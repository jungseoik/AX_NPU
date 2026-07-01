# deploy/vllm — Docker + vLLM(Mobilint NPU) 서빙

`vllm serve`(OpenAI 호환 API)를 **Mobilint NPU**에서 docker compose로 띄운다.
GPU/vLLM 블로그 구성을 NPU로 치환한 것 — 핵심은 **`vllm-mblt` 플러그인**이 vLLM에 `device=cpu`로
등록하고 실제 연산을 qbruntime(NPU)로 보내는 것. 그래서 CUDA/GPU 없이 이 CPU+NPU 서버에서 동작한다.

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

## 실행
```bash
# 레포 루트에서
docker compose -f deploy/vllm/docker-compose.yml --env-file deploy/vllm/.env up -d --build
docker logs -f vllm_mblt          # "Uvicorn running on http://0.0.0.0:8000" 뜨면 성공
curl http://localhost:8000/v1/models
```

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

## ⚠️ 버전 호환 주의 (빌드 전 확인)
- `vllm-mblt`는 **`mblt-model-zoo[transformers]>=1.5.1`** 을 요구 → 이게 **qbruntime 1.2.0보다 최신 런타임**을 요구할 수 있다.
  (우리 검증 조합은 mblt-model-zoo 1.3.1 + qbruntime 1.2.0.) 빌드/기동 시 런타임 버전 에러가 나면
  **최신 qbruntime로 교체**해야 한다 → `download/`에 새 qbruntime tar 두고 Dockerfile `QBRUNTIME_TAR` 조정.
- `vllm==0.11.2` 고정(vllm-mblt가 핀). Python 3.10~3.12.
- 로컬 clone된 `vllm-mblt/`를 쓰려면 Dockerfile의 `pip install "vllm-mblt"` → `COPY vllm-mblt /src && pip install -e /src`로 교체.

## 벤치마크
```bash
docker exec vllm_mblt vllm bench serve --model mobilint/Qwen3-VL-2B-Instruct \
  --trust-remote-code --port 8000 --dataset-name sonnet --dataset-path /workspace/sonnet.txt --num-prompts 10
```
(sonnet.txt는 vllm-mblt 레포에 포함.)

*참고: `vllm-mblt/` = 별개 clone(upstream). 이 deploy는 그걸 pip로 설치해 서빙.*
