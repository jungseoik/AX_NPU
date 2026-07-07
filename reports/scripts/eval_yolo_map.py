"""YOLO NPU mAP 검증 — NPU(mxq) vs fp32(onnxruntime) 동일 val2017 부분집합, pycocotools.
실행: python eval_map.py <mxq> <onnx> <val2017_dir> <ann_json> [N]
env: pe_npu_host (qbruntime + onnxruntime + pycocotools)."""
import sys, os, glob, json
sys.path.insert(0, "/home/gpuadmin/AX_NPU")
import numpy as np, cv2
from yolo_npu.detect import preprocess, postprocess

MXQ, ONNX, VALDIR, ANN = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 300

# COCO 80(연속) → 91(원본 category_id) 매핑
C80_91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,
          35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
          64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

imgs = sorted(glob.glob(os.path.join(VALDIR, "*.jpg")))[:N]
img_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in imgs]


def to_coco(dets, image_id):
    out = []
    for x1, y1, x2, y2, cf, c in dets:
        out.append({"image_id": image_id, "category_id": C80_91[c],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": cf})
    return out


def run_npu():
    import qbruntime
    acc = qbruntime.Accelerator(0); m = qbruntime.Model(MXQ); m.launch(acc)
    res = []
    for p, iid in zip(imgs, img_ids):
        x, r, pad = preprocess(cv2.imread(p))
        o = m.infer(x); o = o[0] if isinstance(o, (list, tuple)) else o
        res += to_coco(postprocess(o, r, pad, conf_thres=0.001, iou_thres=0.7), iid)
    m.dispose(); return res


def run_fp32():
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    res = []
    for p, iid in zip(imgs, img_ids):
        x, r, pad = preprocess(cv2.imread(p))                 # (640,640,3)[0,1]
        inp = np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # (1,3,640,640)
        o = sess.run(None, {iname: inp})[0]                    # (1,84,8400)
        o = np.ascontiguousarray(o.transpose(0, 2, 1))         # (1,8400,84)
        res += to_coco(postprocess(o, r, pad, conf_thres=0.001, iou_thres=0.7), iid)
    return res


def coco_map(results, tag):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        print(f"[{tag}] 검출 0개"); return
    gt = COCO(ANN)
    dt = gt.loadRes(results)
    e = COCOeval(gt, dt, "bbox")
    e.params.imgIds = img_ids
    e.evaluate(); e.accumulate(); e.summarize()
    print(f"[{tag}] mAP@0.5:0.95 = {e.stats[0]:.4f}   mAP@0.5 = {e.stats[1]:.4f}")
    return e.stats[0], e.stats[1]


print(f"평가 이미지 {len(imgs)}장\n=== NPU (mxq, INT8) ===", flush=True)
npu = coco_map(run_npu(), "NPU")
print("\n=== fp32 baseline (onnxruntime) ===", flush=True)
fp = coco_map(run_fp32(), "fp32")
if npu and fp:
    print(f"\n>>> 양자화 손실: mAP@0.5:0.95 {fp[0]:.4f}(fp32) → {npu[0]:.4f}(NPU), "
          f"Δ={npu[0]-fp[0]:+.4f} ({(npu[0]-fp[0])/fp[0]*100:+.1f}%)")
print("EVAL_DONE", flush=True)
