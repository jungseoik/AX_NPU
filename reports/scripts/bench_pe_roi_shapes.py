"""ROI가 어떤 형태로 들어오느냐에 따른 크롭 비용. UI가 빈 리스트가 아니라 사각형을 보내는 경우 포함."""
import os, sys, time, statistics as st
import numpy as np, cv2
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager, _np_crop_region
md = st.median; CH = 62

def ups(n, pts):
    return [{"user_param":{"cameraId":f"c{i}","retEvent":{"falldown_ret":{"roi":{"polygonCoordinates":list(pts)}}}}} for i in range(n)]

def timed(fn, it=5):
    fn(); v=[]
    for _ in range(it):
        s=time.perf_counter(); fn(); v.append((time.perf_counter()-s)*1e3)
    return md(v)

def main():
    rng = np.random.default_rng(0)
    for (W,H) in [(1280,720),(1920,1080)]:
        frames=[rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(CH)]
        cx,cy,rx,ry = W//2,H//2,int(W*0.35),int(H*0.35)
        oct_pts=[]
        for k in range(8):
            a=2*np.pi*k/8; oct_pts += [int(cx+rx*np.cos(a)), int(cy+ry*np.sin(a))]
        cases = {
            "빈 리스트(미지정)":      [],
            "전체화면 사각형(4점)":   [0,0, 0,H, W,H, W,0],
            "부분 사각형(4점)":       [W//4,H//4, W//4,3*H//4, 3*W//4,3*H//4, 3*W//4,H//4],
            "8각형":                  oct_pts,
        }
        print(f"--- {W}x{H}, {CH}채널")
        for name, pts in cases.items():
            roi = PERoIManager()
            u = ups(CH, pts)
            t = timed(lambda: roi.process_batches_with_roi(frames, u))
            c = roi.process_batches_with_roi(frames, u)
            print(f"  {name:22s} crop shape={str(tuple(c[0].shape)):20s} ROI단계 {t:7.1f} ms")
        # 사각형 fast-path 프로토타입 (마스킹 생략, 슬라이스만)
        rect = np.array([0,0, 0,H, W,H, W,0]).reshape(-1,2)
        a = _np_crop_region(frames[0], rect)
        y0,y1,x0,x1 = 0,H-1,0,W-1
        b = frames[0][y0:y1+1, x0:x1+1]
        print(f"  전체화면 사각형: fillPoly 결과 == 슬라이스 결과 ? {np.array_equal(a,b)}")
if __name__ == "__main__":
    main()
