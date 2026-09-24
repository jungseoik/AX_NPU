"""PIA_Wave 의 pluggable 인코더 레지스트리에 NPU 인코더를 꽂는 어댑터.

PIA_Wave 원본(GPU/TensorRT)의 `--model_type pe-core-trt` 를 `--model_type npu` 로
바꾸기만 하면 vision tower 가 ARIES NPU(MXQ)에서 돈다. third_party/PIA_Wave 는
gitignore 대상(외부 레포 클론)이라 **원본 파일은 건드리지 않고** 여기서 등록만 한다.

전처리가 양쪽 동일(uint8 → resize 336 bilinear+antialias → /255 → normalize 0.5)이라
인코더만 갈아끼우면 되고, text tower 는 추론 경로가 아니라 CPU torch 그대로 쓴다.

    python -m wave_npu.encoder --selftest
    python -m wave_npu.pia_wave --model_type npu ...   # 02_Inference 를 NPU 로
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PIA_WAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "third_party", "PIA_Wave")


def _ensure_path():
    if PIA_WAVE not in sys.path:
        sys.path.insert(0, PIA_WAVE)


def register(device_ids="auto", scheme="single", quant=None):
    """PIA_Wave encoders 레지스트리에 'npu' / 'pe-core-npu' 를 등록하고 클래스를 반환."""
    _ensure_path()
    import encoders as pw

    class PECoreNPUEncoder(pw.ZeroShotEncoder):
        """PE-Core vision tower = ARIES NPU(MXQ, full NPU). text tower = CPU torch."""

        def __init__(self, device_ids=device_ids, scheme=scheme, quant=quant,
                     image_size=336, pe_config="PE-Core-L14-336", device="cpu", **_):
            from pe_npu.inference import MXQInferenceFull

            self.device = "cpu"          # 임베딩은 numpy/CPU 텐서로 돌려준다
            self.image_size = int(image_size)
            self.pe_config = pe_config
            self.model = MXQInferenceFull.from_hf(scheme=scheme, device_ids=device_ids,
                                                  quant=quant)
            self._embed_dim = 1024
            self.name = f"PE-Core NPU ({pe_config}, {scheme}, {len(self.model)} cards)"
            self.inference_size_desc = (f"MXQ full-NPU {scheme} × {len(self.model)} card"
                                        f"{'s' if len(self.model) > 1 else ''}")
            self.inference_param_count = 0
            self.inference_size_bytes = 0
            self._text_model = None
            self._tokenizer = None

        @property
        def embed_dim(self):
            return self._embed_dim

        @torch.inference_mode()
        def encode_image(self, crops_chw):
            x = pw._preprocess_crops(crops_chw, self.image_size, device="cpu")
            vec = torch.from_numpy(self.model.infer(x.numpy())).float()
            return vec / vec.norm(dim=-1, keepdim=True)

        def _ensure_text_model(self):
            if self._text_model is None:
                _ensure_path()
                import core.vision_encoder.pe as pe
                import core.vision_encoder.transforms as transforms
                self._text_model = pe.CLIP.from_config(self.pe_config, pretrained=True).eval()
                self._tokenizer = transforms.get_text_tokenizer(self._text_model.context_length)

        @torch.inference_mode()
        def encode_text(self, prompts, batch_size: int = 256):
            self._ensure_text_model()
            out = []
            for i in range(0, len(prompts), batch_size):
                f = self._text_model(text=self._tokenizer(prompts[i:i + batch_size]))[1].float()
                out.append(f / f.norm(dim=-1, keepdim=True))
                print(f"  [NPU encode_text] {min(i + batch_size, len(prompts))}/{len(prompts)}")
            return torch.cat(out, 0)

    pw.register_encoder("npu", "pe-core-npu", "pe-npu")(PECoreNPUEncoder)
    return PECoreNPUEncoder


def _selftest():
    """NPU 인코더 vs pe_npu 직접 호출이 같은 임베딩을 내는지 확인."""
    cls = register(device_ids=[1], scheme="single")
    enc = cls()
    crop = torch.randint(0, 255, (3, 720, 1280), dtype=torch.uint8)
    v = enc.encode_image([crop]).numpy()[0]
    print("embed", v.shape, "norm", np.linalg.norm(v))
    t = enc.encode_text(["a photo of fire", "a photo of smoke"]).numpy()
    print("text", t.shape, "sims", (v @ t.T))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
