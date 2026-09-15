"""사각형 fast-path 정합성: 마스킹 경로와 비트 동일한가 + 비사각형은 영향 없는가."""
import os, sys, time, statistics as st
import numpy as np, cv2, torch
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))
import pia_prod.AI.modules.pe_npu.roi_manager as R
md = st.median

def mask_path(frame, poly, pad=(114,114,114)):
    """fast-path 없이 항상 fillPoly를 타는 참조 구현."""
    h, w = frame.shape[:2]; poly = np.asarray(poly, dtype=np.float32)
    x0=int(np.floor(max(0.,poly[:,0].min()))); y0=int(np.floor(max(0.,poly[:,1].min())))
    x1=int(np.ceil(min(float(w-1),poly[:,0].max()))); y1=int(np.ceil(min(float(h-1),poly[:,1].max())))
    crop = frame[y0:y1+1, x0:x1+1]
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly-(x0,y0)).astype(np.int32)], 255)
    if m.all(): return crop
    out = crop.copy(); out[m==0] = pad[:frame.shape[2]]; return out

def main():
    rng = np.random.default_rng(0)
    for (W,H) in [(1280,720),(1920,1080)]:
        f = rng.integers(0,255,(H,W,3),dtype=np.uint8)
        cx,cy,rx,ry = W//2,H//2,int(W*0.35),int(H*0.35)
        oct_pts=np.array([[int(cx+rx*np.cos(2*np.pi*k/8)), int(cy+ry*np.sin(2*np.pi*k/8))] for k in range(8)])
        cases = {
          "전체화면 사각형": np.array([[0,0],[0,H],[W,H],[W,0]]),
          "부분 사각형":     np.array([[W//4,H//4],[W//4,3*H//4],[3*W//4,3*H//4],[3*W//4,H//4]]),
          "회전 사각형(비축)": np.array([[W//2,H//8],[W//8,H//2],[W//2,7*H//8],[7*W//8,H//2]]),
          "8각형":           oct_pts,
        }
        for name, poly in cases.items():
            a = R._np_crop_region(f, poly); b = mask_path(f, poly)
            rect = R._is_axis_aligned_rect(np.asarray(poly, dtype=np.float32))
            print(f"  {W}x{H} {name:16s} 사각형판정={str(rect):5s} shape={str(a.shape)==str(b.shape):5} 비트동일={np.array_equal(a,b)}")
if __name__ == "__main__":
    main()
