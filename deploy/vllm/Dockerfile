# vLLM + Mobilint NPU (vllm-mblt) 서빙 이미지.
# GPU/CUDA 아님 — vllm-mblt 플랫폼이 vLLM에 device=cpu로 등록하고, 실제 연산은 qbruntime(NPU)로 간다.
# 컨테이너는 host의 /dev/aries* 를 매핑받아 쓴다(드라이버는 host 커널에 설치되어 있어야 함).
#
# 빌드 전제: build context(deploy/vllm의 상위=레포루트)에 download/qbruntime_*_amd64.tar.gz 존재.
#   없으면: python setup/fetch_sdk_from_hf.py 로 먼저 받기 (HF 로그인 필요).
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- qbruntime(런타임 라이브러리 + mobilint-cli) 설치 ---
# download/의 amd64 런타임 tar를 복사해 make install (README: sudo make install)
ARG QBRUNTIME_TAR=download/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz
COPY ${QBRUNTIME_TAR} /tmp/qbruntime.tar.gz
RUN mkdir -p /opt/qbruntime && tar -xzf /tmp/qbruntime.tar.gz -C /opt/qbruntime --strip-components=1 \
    && (cd /opt/qbruntime && make install || true) \
    && rm -f /tmp/qbruntime.tar.gz
# 런타임 lib 경로 (make install이 표준 경로에 안 깔면 대비)
ENV LD_LIBRARY_PATH=/opt/qbruntime/qbruntime/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# --- vLLM + Mobilint 플러그인 ---
# vllm-mblt가 vllm==0.11.2 + mblt-model-zoo[transformers]>=1.5.1 을 끌어온다.
# torch는 CPU 빌드(플랫폼=cpu). vllm-mblt는 PyPI 배포본 사용(로컬 소스 쓰려면 -e 로 교체).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "torch" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "mblt-model-zoo[transformers]==1.5.1" \
    && pip install --no-cache-dir "vllm-mblt"
# 검증: mblt-model-zoo 1.5.1 / mobilint-qb-runtime==1.2.0(PyPI 최신=1.2.0, 우리 네이티브 tar와 일치) /
#       vllm==0.11.2(vllm-mblt가 핀). 버전 충돌 없음 확인됨.

# --- vllm-mblt 0.1.0 버그 패치: VLM(Qwen3-VL) config는 vocab_size가 text_config 안에 있음 ---
# mblt_worker의 self.model.config.vocab_size 가 AttributeError → text_config로 폴백.
# (upstream 보고 대상. 텍스트 모델엔 영향 없음: top-level vocab_size 있으면 그대로 사용.)
RUN python - <<'PY'
p='/usr/local/lib/python3.11/site-packages/vllm_mblt/mblt_worker.py'
s=open(p).read()
old='self.model.config.vocab_size'
new='(getattr(self.model.config, "vocab_size", None) or self.model.config.get_text_config().vocab_size)'
n=s.count(old); s=s.replace(old,new); open(p,'w').write(s)
print(f'[patch] vocab_size fallback applied to {n} site(s)')
assert n>0, 'patch target not found — vllm-mblt 버전 변경 시 확인 필요'
PY

# 헬스체크용
RUN pip install --no-cache-dir requests

EXPOSE 8000
# 실제 실행 명령은 docker-compose.yml의 command에서 (모델·옵션 .env로 주입)
ENTRYPOINT ["vllm"]
