"""
경량 ByteTrack — YOLO NPU 검출 결과를 프레임 간 연결해 track ID 부여 (자체구현, 의존성 numpy+scipy).

트래킹 연산(칼만+IoU+헝가리안)은 CPU로 충분(검출은 NPU가 처리). ByteTrack 방식:
고신뢰 검출로 1차 매칭 → 남은 트랙을 저신뢰 검출로 2차 매칭(가림에 강함).

사용:
    from yolo_npu import YOLONPU, ByteTrack
    det = YOLONPU("yolo11m_single.mxq"); trk = ByteTrack(fps=30)
    # 프레임마다:
    boxes = det(frame)                                    # [(x1,y1,x2,y2,conf,cls),...]
    tracks = trk.update(boxes)                            # [(x1,y1,x2,y2,track_id,conf,cls),...]
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------- bbox 유틸 ----------
def _tlbr_to_xyah(b):
    x1, y1, x2, y2 = b[:4]
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], np.float32)


def _xyah_to_tlbr(x):
    cx, cy, a, h = x[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def _iou_matrix(atlbr, btlbr):
    if len(atlbr) == 0 or len(btlbr) == 0:
        return np.zeros((len(atlbr), len(btlbr)), np.float32)
    a = np.asarray(atlbr, np.float32)[:, None, :]   # (N,1,4)
    b = np.asarray(btlbr, np.float32)[None, :, :]   # (1,M,4)
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-6)


# ---------- 칼만 필터 (ByteTrack 표준, 상태 [cx,cy,a,h,v...]) ----------
class _Kalman:
    def __init__(self):
        n, dt = 4, 1.0
        self._F = np.eye(2 * n)
        for i in range(n):
            self._F[i, n + i] = dt
        self._H = np.eye(n, 2 * n)
        self._sp = 1.0 / 20    # position std weight
        self._sv = 1.0 / 160   # velocity std weight

    def initiate(self, m):
        mean = np.r_[m, np.zeros(4, np.float32)]
        std = [2 * self._sp * m[3], 2 * self._sp * m[3], 1e-2, 2 * self._sp * m[3],
               10 * self._sv * m[3], 10 * self._sv * m[3], 1e-5, 10 * self._sv * m[3]]
        return mean, np.diag(np.square(std)).astype(np.float32)

    def predict(self, mean, cov):
        std = [self._sp * mean[3], self._sp * mean[3], 1e-2, self._sp * mean[3],
               self._sv * mean[3], self._sv * mean[3], 1e-5, self._sv * mean[3]]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean, cov, m):
        std = [self._sp * mean[3], self._sp * mean[3], 1e-1, self._sp * mean[3]]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        mean = mean + K @ (m - self._H @ mean)
        cov = (np.eye(8) - K @ self._H) @ cov
        return mean, cov


_KF = _Kalman()


class _Track:
    _count = 0

    def __init__(self, tlbr, score, cls):
        self.mean, self.cov = _KF.initiate(_tlbr_to_xyah(tlbr))
        self.score = score
        self.cls = cls
        self.track_id = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = "new"          # new/tracked/lost

    def activate(self):
        _Track._count += 1
        self.track_id = _Track._count
        self.state = "tracked"

    def predict(self):
        self.mean, self.cov = _KF.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, tlbr, score, cls):
        self.mean, self.cov = _KF.update(self.mean, self.cov, _tlbr_to_xyah(tlbr))
        self.score = score
        self.cls = cls
        self.hits += 1
        self.time_since_update = 0
        self.state = "tracked"

    @property
    def tlbr(self):
        return _xyah_to_tlbr(self.mean)


class ByteTrack:
    """경량 ByteTrack. update(dets) -> [(x1,y1,x2,y2,track_id,score,cls), ...]."""

    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, min_box_area=10, fps=30):
        self.track_thresh = track_thresh        # 고/저신뢰 경계
        self.match_thresh = match_thresh         # IoU 매칭 임계(=1-IoU 거리)
        self.max_age = int(track_buffer * fps / 30)   # lost 유지 프레임
        self.min_box_area = min_box_area
        self.tracks: list[_Track] = []

    @staticmethod
    def _match(tracks, dets_tlbr, iou_thresh):
        """IoU 헝가리안 매칭. 반환: (matches[(ti,di)], un_tracks, un_dets)."""
        if not tracks or len(dets_tlbr) == 0:
            return [], list(range(len(tracks))), list(range(len(dets_tlbr)))
        iou = _iou_matrix([t.tlbr for t in tracks], dets_tlbr)
        cost = 1.0 - iou
        ti, di = linear_sum_assignment(cost)
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(dets_tlbr)))
        for r, c in zip(ti, di):
            if iou[r, c] >= iou_thresh:
                matches.append((r, c)); ut.discard(r); ud.discard(c)
        return matches, list(ut), list(ud)

    def update(self, detections):
        dets = np.asarray([d[:6] for d in detections], np.float32).reshape(-1, 6)
        scores = dets[:, 4]
        high = dets[scores >= self.track_thresh]
        low = dets[(scores < self.track_thresh) & (scores >= 0.1)]

        for t in self.tracks:
            t.predict()

        # 1차: 모든 트랙 ↔ 고신뢰 검출
        m1, ut1, ud1 = self._match(self.tracks, high[:, :4], self.match_thresh)
        for ti, di in m1:
            self.tracks[ti].update(high[di, :4], high[di, 4], int(high[di, 5]))

        # 2차: 남은 (tracked였던) 트랙 ↔ 저신뢰 검출
        rem = [self.tracks[i] for i in ut1]
        m2, ut2, _ = self._match(rem, low[:, :4], 0.5)
        for ti, di in m2:
            rem[ti].update(low[di, :4], low[di, 4], int(low[di, 5]))

        # 매칭 안 된 트랙 → lost, 오래되면 제거
        lost_idx = [ut1[i] for i in ut2]
        for i in lost_idx:
            self.tracks[i].state = "lost"
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 매칭 안 된 고신뢰 검출 → 신규 트랙 활성화
        for di in ud1:
            b = high[di]
            if (b[2] - b[0]) * (b[3] - b[1]) < self.min_box_area:
                continue
            t = _Track(b[:4], b[4], int(b[5])); t.activate()
            self.tracks.append(t)

        # 출력: 현재 프레임에 매칭된(tracked) 트랙만
        out = []
        for t in self.tracks:
            if t.state == "tracked" and t.time_since_update == 0:
                x1, y1, x2, y2 = t.tlbr
                out.append((float(x1), float(y1), float(x2), float(y2), t.track_id, float(t.score), t.cls))
        return out

    @staticmethod
    def reset_ids():
        _Track._count = 0


def draw_tracks(img_bgr, tracks, names=None):
    """track 리스트 [(x1,y1,x2,y2,track_id,score,cls),...]를 이미지에 그림(ID별 색). BGR 반환."""
    import cv2
    img = img_bgr.copy()
    for x1, y1, x2, y2, tid, score, cls in tracks:
        rng = np.random.default_rng(int(tid) * 9973 + 7)     # ID별 고정 색
        col = [int(v) for v in rng.integers(64, 256, 3)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, col, 2)
        name = names[cls] if names and cls < len(names) else str(cls)
        label = f"#{int(tid)} {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), col, -1)
        cv2.putText(img, label, (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img
