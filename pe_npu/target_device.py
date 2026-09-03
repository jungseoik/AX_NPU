"""
컴파일 타겟 디바이스 이름 해석.

**같은 하드웨어(Aries2)인데 qbcompiler 버전에 따라 이름이 다르다.**

| qbcompiler | 받는 이름 | 비고 |
| --- | --- | --- |
| 1.1.x (SDK 1.0v) | `aries2` | `aries-rb`를 모른다 |
| 1.2.x (SDK 1.1v) | `aries-rb` | `aries2`를 거부한다(`ValueError: Unsupported device name`) |

1.2.0의 선택지는 `regulus-ra` / `aries-rb` / `regulus-rb` 이고, 이 중 **`aries-rb`가 Aries2**다.
실제로 `aries-rb`로 컴파일한 mxq를 `mxqtool show`로 보면 `Hardware Version: Aries2`,
`Format Version: 0x70000`(v7)으로 1.1.2 산출물과 동일하다. 즉 **산출물 호환성은 그대로**이고
드라이버 1.13 / 런타임 1.2.0 환경에서 그대로 로드·추론된다.
→ `reports/performance/NPU_pe_quant_tuning_compiler_version.md`

그래서 코드에 이름을 하드코딩하면 컴파일러를 올린 순간 깨진다. 아래 우선순위로 해석한다.

1. 명시 인자 (`--target-device`)
2. 환경변수 `AX_NPU_TARGET_DEVICE`
3. 설치된 `qbcompiler.__version__` 자동 감지  ← 기본 경로
4. `aries2` 폴백
"""
from __future__ import annotations

import os

#: qbcompiler (major, minor) 하한 → Aries2를 가리키는 타겟 이름
_ARIES2_NAME_BY_COMPILER = (
    ((1, 2), "aries-rb"),   # 1.2.0+ : aries2 이름 폐기
    ((0, 0), "aries2"),     # 그 이전
)

FALLBACK = "aries2"
ENV_VAR = "AX_NPU_TARGET_DEVICE"


def compiler_version() -> tuple[int, int] | None:
    """설치된 qbcompiler의 (major, minor). 조회 실패 시 None."""
    try:
        import qbcompiler
        raw = getattr(qbcompiler, "__version__", None)
        if raw is None:
            import importlib.metadata as md
            raw = md.version("qbcompiler")
        major, minor = (int(x) for x in str(raw).split(".")[:2])
        return major, minor
    except Exception:
        return None


def resolve_target_device(explicit: str | None = None, *, verbose: bool = True) -> str:
    """Aries2용 target_device 이름을 해석한다.

    explicit 이 주어지면 그대로 쓴다(사용자가 regulus 등 다른 칩을 지정할 수도 있다).
    """
    if explicit:
        return explicit

    env = os.environ.get(ENV_VAR)
    if env:
        if verbose:
            print(f"[target] {ENV_VAR}={env}")
        return env

    ver = compiler_version()
    if ver is None:
        if verbose:
            print(f"[target] qbcompiler 버전 조회 실패 → {FALLBACK}")
        return FALLBACK

    for minimum, name in _ARIES2_NAME_BY_COMPILER:
        if ver >= minimum:
            if verbose:
                print(f"[target] qbcompiler {ver[0]}.{ver[1]} → target_device={name}")
            return name
    return FALLBACK


def add_argument(parser) -> None:
    """CLI에 --target-device 를 붙인다(모든 컴파일 CLI 공통)."""
    parser.add_argument(
        "--target-device", default=None,
        help="컴파일 타겟 칩 이름. 미지정 시 qbcompiler 버전으로 자동 판별 "
             "(1.1.x→aries2 / 1.2.x→aries-rb, 둘 다 Aries2 하드웨어). "
             f"환경변수 {ENV_VAR} 로도 지정 가능",
    )
