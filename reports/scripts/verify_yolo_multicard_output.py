"""멀티카드 출력 무결성 검증: 서로 다른 이미지 N장을 단일카드 vs 2장/7장 분산 비교.
각 위치(이미지)의 검출이 정확히 일치해야 함(순서보존 + 카드간 오염 없음)."""
import sys, glob
sys.path.insert(0, "/home/gpuadmin/AX_NPU")
import numpy as np
from yolo_npu import YOLONPU, detect_npu_devices

MXQ = "/home/gpuadmin/AX_NPU/scratch_yolo/final/yolo11m_single.mxq"
imgs = sorted(glob.glob("/home/gpuadmin/AX_NPU/scratch_yolo/val2017/*.jpg"))[:40]
print(f"서로 다른 이미지 {len(imgs)}장, NPU {detect_npu_devices()}\n", flush=True)


def det_eq(a, b, tol=1e-3):
    """두 검출 리스트가 같은지 (개수 + 클래스 + 좌표/conf)."""
    if len(a) != len(b):
        return False
    for da, db in zip(a, b):
        if da[5] != db[5]:
            return False
        if not np.allclose(da[:5], db[:5], atol=tol):
            return False
    return True


# 기준: 단일카드(aries0)에서 한 장씩
ref_det = YOLONPU(MXQ, device_id=0)
ref = [ref_det(p) for p in imgs]
del ref_det
print(f"기준(단일카드) 완료. 총 검출 {sum(len(r) for r in ref)}개", flush=True)

for k in [2, 7]:
    ids = detect_npu_devices()[:k]
    mc = YOLONPU(MXQ, device_ids=ids)
    out = mc.detect_batch(imgs)                      # 카드 분산(라운드로빈+스레드)
    mism = [i for i in range(len(imgs)) if not det_eq(out[i], ref[i])]
    # 클래스 시퀀스 몇 개 샘플로 실제 다양성 확인(같은 결과 아님을 증명)
    div = len(set(tuple(sorted(d[5] for d in r)) for r in ref))
    print(f"[{k}장 {ids}] 불일치 {len(mism)}/{len(imgs)}  "
          f"{'✅ 완전일치(오염 없음)' if not mism else '❌ 불일치 idx='+str(mism[:10])}  "
          f"(이미지별 클래스조합 {div}종 = 서로 다른 이미지 확인)", flush=True)
    del mc
print("VERIFY_DONE", flush=True)
