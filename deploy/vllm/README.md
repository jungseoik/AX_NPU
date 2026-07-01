# deploy/vllm — Docker + vLLM(Mobilint NPU) 서빙

`vllm serve`(OpenAI 호환 API)를 **Mobilint NPU**에서 docker compose로 띄운다.
GPU/vLLM 블로그 구성을 NPU로 치환한 것 — 핵심은 **`vllm-mblt` 플러그인**이 vLLM에 `device=cpu`로
등록하고 실제 연산을 qbruntime(NPU)로 보내는 것. 그래서 CUDA/GPU 없이 이 CPU+NPU 서버에서 동작한다.

> **운영 기준 = batch 1.** 검증 모델 `mobilint/Qwen3-VL-2B-Instruct`는 **batch=1로 컴파일된 MXQ**라
> 동시요청은 vLLM이 직렬 큐잉한다(NPU 메모리 ~2.5GB 고정, 부하로 안 터짐 — `loadtest_vllm.py` 실측).
> **배치>1이 필요하면** batch-compiled MXQ가 있어야 하고, 그건 **Mobilint에 요청**해야 한다
> (실행중 `--model-loader-extra-config`로 batch 올리기는 Qwen3-VL에서 불가 — 아래 [한계] 참조).
> 참고 clone(추적 안 함): `vllm-mblt`(플러그인 소스), `mblt-model-zoo`(런타임 래퍼), `mblt-sdk-tutorial`(컴파일 레시피).

## GPU 블로그 vs 여기(NPU) 차이
| GPU 블로그 | 여기 (NPU) |
|---|---|
| `image: vllm/vllm-openai`(CUDA) | 커스텀 이미지: qbruntime + `vllm-mblt` |
| `runtime: nvidia` + gpu reservation | **`devices: /dev/aries0~6`** 매핑 |
| `--gpu-memory-utilization` | 불필요 (NPU는 worker가 관리) |
| `Qwen/Qwen3-VL-*`(GPU 가중치) | **`mobilint/Qwen3-VL-*`**(NPU MXQ) |
| — | `--model-loader-extra-config`로 core_mode/dev_no 등 |

## 사전 준비
1. **host에 NPU 드라이버 설치** + `/dev/aries*` 존재 (`.claude/skills/npu-setup` 로 세팅).
2. **qbruntime tar가 `download/`에 있어야** (이미지 빌드 시 COPY). 없으면:
   `python setup/fetch_sdk_from_hf.py` (HF 로그인 필요).
3. **Docker + docker compose** 설치.
4. `.env` 작성: `cp deploy/vllm/.env.example deploy/vllm/.env` 후 `HF_TOKEN`/`MODEL_NAME` 채우기.

## 실행 (검증됨 ✅)
```bash
# 레포 루트에서
docker compose -f deploy/vllm/docker-compose.yml --env-file deploy/vllm/.env up -d --build
docker logs -f vllm_mblt          # "Application startup complete." 뜨면 성공
curl http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl","messages":[{"role":"user","content":"안녕"}],"max_tokens":50}'
# 중지: docker compose -f deploy/vllm/docker-compose.yml down
```
> **실측 확인**(2026-07, 이 서버 CPU+7×ARIES): Qwen3-VL-2B가 device_config=cpu + NPU(qbruntime)로
> `from_pretrained` 3.8s에 로드, `Application startup complete`, `/v1/chat/completions` 정상 응답.

### 기동 시 주의(실전 팁)
- **`MAX_MODEL_LEN`은 모델 config의 `max_position_embeddings` 이하**로. Qwen3-VL-2B=**4096**. 초과 시 기동 실패.
- `command`는 **list 형식**으로 둠(JSON 인자 quoting 문제 회피). **기본은 config.json 레이아웃(batch1)**을 따르며
  `--model-loader-extra-config`를 **넣지 않는다**(Qwen3-VL에선 이게 크래시 유발 — [한계] 참조. 텍스트 batch 모델에서만 opt-in).
- vllm이 CUDA torch/nvidia 휠을 끌어와 이미지가 큼(~11GB). GPU 없어도 `libcuda.so.1 ... cannot open`,
  `Triton ... 0 driver` **경고는 정상**(mblt=cpu 경로라 무해).

### API 테스트 (멀티모달, OpenAI 클라이언트)
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = c.chat.completions.create(model="qwen3-vl", messages=[{"role":"user","content":[
    {"type":"text","text":"이 이미지 설명해줘"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}]}], max_tokens=512)
print(r.choices[0].message.content)
```
> Mobilint Qwen3-VL NPU 경로: 초기 요청에 **이미지 1장**, 이후 턴은 텍스트 전용/같은 이미지 위치 재사용, **비디오 미지원**.

## 모델 바꾸기
`.env`의 `MODEL_NAME` 한 줄만 교체 → `docker compose ... up -d`. (텍스트: `mobilint/Llama-3.2-1B-Instruct` 등)

## 버전 호환 (확인 완료 ✅)
- `vllm-mblt 0.1.0` → `mblt-model-zoo[transformers]>=1.5.1` → **`mobilint-qb-runtime>=1.2.0`**.
- PyPI `mobilint-qb-runtime` **최신 = 1.2.0** → `>=1.2.0`은 항상 1.2.0으로 해석 = **우리 네이티브 qbruntime tar(1.2.0)와 일치.** 충돌 없음.
- Dockerfile은 `mblt-model-zoo==1.5.1` 핀(2.0.0도 qb-runtime 1.2.0 요구라 무방하나 vllm-mblt 검증 버전에 맞춤).
- `vllm==0.11.2` 고정(vllm-mblt가 핀). Python 3.10~3.12.
- 남은 실전 리스크는 **vllm 0.11.2 CPU 임포트**(CUDA 없는 환경) — 빌드/기동에서 확인. 문제 시 vllm CPU 빌드 경로로 조정.
- 로컬 clone된 `vllm-mblt/`를 쓰려면 Dockerfile의 `pip install "vllm-mblt"` → `COPY vllm-mblt /src && pip install -e /src`로 교체.

## 벤치마크
```bash
docker exec vllm_mblt vllm bench serve --model mobilint/Qwen3-VL-2B-Instruct \
  --trust-remote-code --port 8000 --dataset-name sonnet --dataset-path /workspace/sonnet.txt --num-prompts 10
```
(sonnet.txt는 vllm-mblt 레포에 포함.)

*참고: `vllm-mblt/` = 별개 clone(upstream). 이 deploy는 그걸 pip로 설치해 서빙.*

## [한계] 배치(batch) — 왜 못 올리나 (실측·조사 결론)
- **배치는 MXQ 컴파일 시 박히는 값**(`max_batch_size`). `mobilint/Qwen3-VL-2B`는 batch=1.
- 실행중 override(`--model-loader-extra-config '{"max_batch_size":N}'`)는 **Qwen3-VL에서 불가**:
  dev_no/max_batch_size가 `from_pretrained`로 넘어가 wrapper `__init__` TypeError → 엔진 크래시(실측).
- `mblt-model-zoo`(런타임 래퍼)엔 컴파일러 없음. `mblt-sdk-tutorial`엔 VLM/LLM 컴파일 레시피 있고
  우리 qbcompiler에 `qwen3vl` 파서도 있어 **직접 batch 컴파일은 이론상 가능**하나(Qwen2-VL 레시피를
  Qwen3-VL로 포팅 + vision/language 각각 컴파일 + 패키징, ~20GB·수 시간), 난이도 높음.
- **권장: Mobilint에 batch-compiled Qwen3-VL-2B MXQ 요청**(`Llama-3.2-1B-Instruct-Batch32` 선례).
  받으면 `.env`의 `MODEL_NAME`만 교체해 이 구성 그대로 배치 서빙 가능.

## 알려진 이슈(vllm-mblt 0.1.0, Mobilint 보고 대상)
1. VLM(Qwen3-VL) `config.vocab_size` AttributeError → Dockerfile에서 `text_config` 폴백 패치 적용.
2. Qwen3-VL에 `--model-loader-extra-config`(dev_no 등) 전달 시 `from_pretrained` TypeError.
