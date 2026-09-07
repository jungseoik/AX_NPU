from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def diagnostic(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def quant_config_builder():
    module = importlib.import_module("pe_npu.compile")
    builder = getattr(module, "_build_bit_config", None)
    assert callable(builder), "pe_npu.compile._build_bit_config is missing"
    return builder


def import_required(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"required module is missing: {module_name} ({exc})")


def test_compile_help_exposes_vendor_quantization_controls() -> None:
    result = run_cli("-m", "pe_npu.compile", "--help")

    assert result.returncode == 0, diagnostic(result)
    for option in (
        "--quant",
        "--optq",
        "--search-weight-scale",
        "--a16",
        "--calib-stats-save",
        "--calib-stats-load",
    ):
        assert option in result.stdout, diagnostic(result)


def test_benchmark_help_accepts_explicit_local_artifacts_and_images() -> None:
    script = ROOT / "reports" / "scripts" / "bench_modes_threaded.py"
    result = run_cli(str(script), "--help")

    assert result.returncode == 0, diagnostic(result)
    for option in (
        "--model",
        "--images-dir",
        "--device",
        "--batch",
        "--threads",
        "--json-out",
    ):
        assert option in result.stdout, diagnostic(result)


@pytest.mark.parametrize(
    ("quant", "activation_output", "weight_bits"),
    [
        ("w8a16", 16, {"query": 8, "key": 8, "value": 8, "output": 8, "ffn": 8, "head": 8}),
        ("w4a16", 16, {"query": 4, "key": 4, "value": 8, "output": 4, "ffn": 4, "head": 4}),
        ("w4a8", 8, {"query": 4, "key": 4, "value": 8, "output": 4, "ffn": 4, "head": 4}),
    ],
)
def test_quant_presets_match_vendor_bits(
    quant: str, activation_output: int, weight_bits: dict[str, int]
) -> None:
    config = quant_config_builder()(
        quant=quant,
        activation_16bits=[],
        weight_16bits=[],
        bit4=0.0,
    ).model_dump()

    activation = config["transformer"]["activation"]
    assert activation == {
        "query": 8,
        "key": 8,
        "value": 8,
        "output": activation_output,
        "ffn": activation_output,
        "head": 8,
    }
    assert config["transformer"]["weight"] == weight_bits


def test_user_a16_and_qk16_names_are_deduplicated_in_one_override() -> None:
    config = quant_config_builder()(
        quant="w4a8",
        activation_16bits=["user/op", "score/qk", "user/op"],
        weight_16bits=["weight/op", "weight/op"],
        bit4=0.0,
    ).model_dump()

    overrides = config["layer_overrides"]
    assert overrides["activation_16bits"] == ["score/qk", "user/op"]
    assert overrides["weight_16bits"] == ["weight/op"]


def test_vendor_preset_and_legacy_mixed_precision_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        quant_config_builder()(
            quant="w4a16",
            activation_16bits=[],
            weight_16bits=[],
            bit4=0.5,
        )


def test_optimizer_configs_match_vendor_parameters() -> None:
    module = importlib.import_module("pe_npu.compile")
    builder = getattr(module, "_build_optimizer_configs", None)
    assert callable(builder), "pe_npu.compile._build_optimizer_configs is missing"

    configs = builder(optq=True, search_weight_scale=True)
    assert configs["optq_config"].model_dump() == {
        "apply": True,
        "attributes": {
            "act_order": True,
            "block_size": 128,
            "perc_damp": 0.01,
            "apply_layers": [],
            "exclude_layers": [],
        },
    }
    assert configs["search_weight_scale_config"].model_dump() == {
        "apply": True,
        "transformer": {
            "query": True,
            "key": True,
            "value": True,
            "out": True,
            "ffn": True,
        },
    }


def test_calibration_statistics_cache_uses_one_matching_percentile() -> None:
    module = importlib.import_module("pe_npu.compile")
    builder = getattr(module, "_build_calibration_config", None)
    assert callable(builder), "pe_npu.compile._build_calibration_config is missing"

    saved = builder(
        method=1,
        output=1,
        stats_save="/tmp/coco200_stats.json",
        stats_load=None,
    ).model_dump()["statistics"]
    assert saved == {
        "apply": True,
        "save_path": "/tmp/coco200_stats.json",
        "load_path": "",
        "percentiles": [0.9999],
        "percentile_index": 0,
    }

    loaded = builder(
        method=1,
        output=1,
        stats_save=None,
        stats_load="/tmp/coco200_stats.json",
    ).model_dump()["statistics"]
    assert loaded == {
        "apply": True,
        "save_path": "",
        "load_path": "/tmp/coco200_stats.json",
        "percentiles": [0.9999],
        "percentile_index": 0,
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        builder(
            method=1,
            output=1,
            stats_save="save.json",
            stats_load="load.json",
        )


def test_optq_rejects_statistics_load_that_skips_required_hessian() -> None:
    module = importlib.import_module("pe_npu.compile")
    validate = getattr(module, "_validate_calibration_strategy", None)
    assert callable(validate), "pe_npu.compile._validate_calibration_strategy is missing"

    with pytest.raises(ValueError, match="OPTQ Hessian"):
        validate(optq=True, stats_load="/tmp/coco200_stats")

    validate(
        optq=False,
        stats_load="/tmp/coco200_stats",
        calib_path="/tmp/calib/npy_files.txt",
    )
    validate(optq=True, stats_load=None)


def test_calibration_statistics_require_explicit_calibration_data() -> None:
    module = importlib.import_module("pe_npu.compile")
    validate = getattr(module, "_validate_calibration_strategy", None)
    assert callable(validate), "pe_npu.compile._validate_calibration_strategy is missing"

    with pytest.raises(ValueError, match="--calib-data-path"):
        validate(
            optq=False,
            stats_save="/tmp/coco200_stats",
            stats_load=None,
            calib_path=None,
        )
    with pytest.raises(ValueError, match="--calib-data-path"):
        validate(
            optq=False,
            stats_save=None,
            stats_load="/tmp/coco200_stats",
            calib_path=None,
        )

    validate(
        optq=False,
        stats_save="/tmp/coco200_stats",
        stats_load=None,
        calib_path="/tmp/calib/npy_files.txt",
    )


def test_calibration_stats_load_output_is_normalized_to_requested_path(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("pe_npu.compile")
    normalize = getattr(module, "_normalize_stats_output", None)
    assert callable(normalize), "pe_npu.compile._normalize_stats_output is missing"

    requested = tmp_path / "pe_w4a16_single.mxq"
    requested.write_bytes(b"stale requested output")
    suffixed = tmp_path / "pe_w4a16_single_0.9999.mxq"
    suffixed.write_bytes(b"fresh compiler output")

    actual = normalize(str(requested), stats_load=True)

    assert actual == str(requested)
    assert requested.read_bytes() == b"fresh compiler output"
    assert not suffixed.exists()


def test_benchmark_rejects_missing_explicit_model(tmp_path: Path) -> None:
    script = ROOT / "reports" / "scripts" / "bench_modes_threaded.py"
    result = run_cli(
        str(script),
        "--model",
        f"single={tmp_path / 'missing.mxq'}",
        "--images-dir",
        str(tmp_path),
        "--batch",
        "1",
    )

    assert result.returncode != 0
    assert "MXQ file does not exist" in result.stderr, diagnostic(result)


def test_benchmark_requires_enough_distinct_images(tmp_path: Path) -> None:
    script = ROOT / "reports" / "scripts" / "bench_modes_threaded.py"
    model = tmp_path / "model.mxq"
    model.write_bytes(b"placeholder")
    result = run_cli(
        str(script),
        "--model",
        f"single={model}",
        "--images-dir",
        str(tmp_path),
        "--batch",
        "1",
    )

    assert result.returncode != 0
    assert "need at least 1 distinct image" in result.stderr, diagnostic(result)


def test_benchmark_integrity_rejects_nonfinite_cosines() -> None:
    module = import_required("reports.scripts.bench_modes_threaded")
    validate = getattr(module, "validate_cosines", None)
    assert callable(validate), "validate_cosines is missing"

    with pytest.raises(RuntimeError, match="non-finite"):
        validate("nan-output", [1.0, float("nan")])
    with pytest.raises(RuntimeError, match="non-finite"):
        validate("inf-output", [1.0, float("inf")])
    with pytest.raises(RuntimeError, match="min_cos"):
        validate("low-cos", [1.0, 0.9])

    assert validate("valid", [1.0, 0.99995]) == pytest.approx(0.99995)


def test_profiler_tracks_parent_attention_as_out_projection() -> None:
    module = import_required("pe_npu.profile_activations")
    label = getattr(module, "profile_label", None)
    assert callable(label), "profile_label is missing"

    assert label("visual.transformer.resblocks.3.attn") == "L3.attn.out_proj"
    assert label("visual.transformer.resblocks.3.attn.out_proj") is None
    assert label("visual.transformer.resblocks.12.mlp.c_proj") == "L12.mlp.c_proj"
    assert label("visual.transformer.resblocks.12.ln_1") == "L12.ln_1"


def test_profiler_summary_reports_max_p999_and_ratio() -> None:
    module = import_required("pe_npu.profile_activations")
    summarize = getattr(module, "summarize_values", None)
    assert callable(summarize), "summarize_values is missing"

    values = np.array([0.0, 1.0, 2.0, 100.0], dtype=np.float32)
    row = summarize(max_abs=100.0, sampled_abs=values, seen=10)
    expected_p999 = float(np.quantile(values, 0.999))
    assert row["max"] == 100.0
    assert row["p99_9"] == pytest.approx(expected_p999)
    assert row["ratio"] == pytest.approx(100.0 / expected_p999)
    assert row["seen"] == 10
    assert row["samples"] == 4


def test_profiler_sampling_avoids_channel_stride_aliasing() -> None:
    module = import_required("pe_npu.profile_activations")
    sample_indices = getattr(module, "sample_indices", None)
    assert callable(sample_indices), "sample_indices is missing"

    first = sample_indices(total=577 * 1024, take=4096, seed=17)
    second = sample_indices(total=577 * 1024, take=4096, seed=17)
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 4096
    assert int(first.min()) >= 0
    assert int(first.max()) < 577 * 1024
    # A regular stride of 144 sees only 64/1024 channel residues.  The
    # deterministic random sample should cover nearly all channels.
    assert len(np.unique(first % 1024)) > 900


def test_cosine_summary_uses_per_image_vectors() -> None:
    module = import_required("reports.scripts.eval_cos_pth_vs_mxq")
    summarize = getattr(module, "cosine_summary", None)
    assert callable(summarize), "cosine_summary is missing"

    reference = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    summary = summarize(reference, candidate)

    assert summary["count"] == 2
    assert summary["mean_cos"] == pytest.approx(0.5)
    assert summary["min_cos"] == pytest.approx(0.0)
    assert summary["max_cos"] == pytest.approx(1.0)
    assert summary["cosines"] == pytest.approx([1.0, 0.0])


def test_cosine_summary_rejects_shape_mismatch() -> None:
    module = import_required("reports.scripts.eval_cos_pth_vs_mxq")
    summarize = getattr(module, "cosine_summary", None)
    assert callable(summarize), "cosine_summary is missing"

    with pytest.raises(ValueError, match="shape mismatch"):
        summarize(np.zeros((2, 4)), np.zeros((1, 4)))


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (np.array([[1.0, 0.0]]), np.array([[np.nan, 0.0]])),
        (np.array([[1.0, 0.0]]), np.array([[np.inf, 0.0]])),
        (np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]])),
    ],
)
def test_cosine_summary_rejects_nonfinite_or_zero_norm_embeddings(
    reference: np.ndarray, candidate: np.ndarray
) -> None:
    module = import_required("reports.scripts.eval_cos_pth_vs_mxq")
    summarize = getattr(module, "cosine_summary", None)
    assert callable(summarize), "cosine_summary is missing"

    with pytest.raises(ValueError, match="non-finite|zero-norm"):
        summarize(reference, candidate)
