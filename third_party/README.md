# third_party/ — 외부 레포 클론

이 레포와 **별개로 관리되는 외부 저장소**를 여기에 clone해서 참조한다.
각자 독립된 git 저장소이고, **AX_NPU git에는 추적되지 않는다**(`.gitignore`로 `third_party/*` 제외,
이 README만 추적). 서브모듈이 아니므로 커밋이 핀으로 박히지 않고, 각 레포에서 직접 pull하면 된다.

| 폴더 | 원본 | 기본 브랜치 | 비고 |
|---|---|---|---|
| `APO-AI-GUI/` | `github.com/TeamPIA/APO-AI-GUI` (private) | `main` | |
| `Product-AI-mono/` | `github.com/jungseoik/Product-AI-mono` (private) | `dev` | 레포 루트의 구 `Product-AI-mono/`(assets 사본)와 별개 |

## 받기 / 갱신

private 레포라 GitHub 토큰이 필요하다(`.env`의 `GITHUB_TOKEN`).

```bash
set -a; . .env; set +a
mkdir -p third_party
for u in TeamPIA/APO-AI-GUI jungseoik/Product-AI-mono; do
  n=$(basename "$u")
  git clone "https://${GITHUB_TOKEN}@github.com/${u}.git" "third_party/$n"
  git -C "third_party/$n" remote set-url origin "https://github.com/${u}.git"   # 토큰 저장 방지
done
```

갱신은 각 폴더에서 평소처럼:

```bash
git -C third_party/APO-AI-GUI pull
git -C third_party/Product-AI-mono pull
```

> `remote set-url`로 토큰을 뺐기 때문에 pull 시 인증을 다시 물어본다.
> 매번 넣기 싫으면 `git config --global credential.helper store` 또는 gh CLI 로그인을 쓸 것.
> **토큰이 박힌 URL을 커밋하지 말 것.**

## 서브모듈(`mblt-model-zoo/`, `mblt-sdk-tutorial/`)과의 차이

| | `third_party/` (일반 clone) | 서브모듈 |
|---|---|---|
| AX_NPU git 추적 | 안 함(gitignore) | 함 — **특정 커밋에 핀** |
| 새 서버에서 clone 시 | 위 스크립트로 직접 받아야 함 | `git submodule update --init` → **핀된 커밋** |
| 최신으로 올리기 | `git pull` (자유) | `git submodule update --remote` 후 **AX_NPU에 커밋해야 반영** |

버전을 재현 가능하게 고정해야 하면 서브모듈, 그냥 최신 소스를 참고용으로 두려면 여기(`third_party/`)가 맞다.
