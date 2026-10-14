# -*- coding: utf-8 -*-
"""2차 판정 검토용 그리드 페이지 생성.

알람 162건을 **2차가 무엇을 걸렀는가** 기준으로 4분류해서 보여준다.

  정탐 기각 ← 이게 핵심. 실제 이벤트인데 2차가 떨어뜨린 건
  정탐 통과
  오탐 기각   2차가 제 역할을 한 건
  오탐 통과   2차가 놓친 건

원본과 **2차가 실제로 본 256x192** 를 나란히 둔다. 사람이 원본만 보고
"이건 명백한 화재인데 왜 기각하냐"고 판단하면 2차를 과소평가하게 된다.
"""
from __future__ import annotations

import base64
import io
import json
import os

import cv2

import settings as S

OUT = os.path.join(S.OUT, "review")


def b64(img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    s1 = json.load(open(os.path.join(S.OUT, "stage1.json"), encoding="utf-8"))
    verd = {r["evidence"]: r for r in json.load(
        open(os.path.join(S.OUT, "stage2.json"), encoding="utf-8"))}

    # 2차 단독(1차·FSM 없이 매 초 전 프레임 VQA) — 있으면 같이 띄운다
    solo = {}
    sp = os.path.join(S.OUT, "stage2_only.json")
    if os.path.exists(sp):
        for c in json.load(open(sp, encoding="utf-8"))["clips"]:
            for cat, pr in c["pred"].items():
                solo[(c["cat_dir"], c["name"], cat)] = pr

    items = []
    for c in s1["clips"]:
        for a in c["alarms"]:
            cat = a["cat"]
            if c["cat_dir"] in S.EVAL_EXCLUDE.get(cat, ()):
                continue                      # 평가 범위 밖(화재 영상의 연기 판정)
            gt = c["gt"][cat]
            clip_pos = sum(gt) > 0
            v = verd.get(a["evidence"], {})
            lab = v.get("label", "?")
            kind = ("정탐 기각" if clip_pos and lab == "no" else
                    "정탐 통과" if clip_pos and lab == "yes" else
                    "오탐 기각" if lab == "no" else "오탐 통과")
            # 2단계 최종 = 2차가 통과시킨 에피소드만 남긴 것. F1 은 이걸 쓴다.
            act = c["active"][cat]
            fin = [0] * len(act)
            if lab == "yes":
                s_ = e_ = a["idx"]
                while s_ > 0 and act[s_ - 1]:
                    s_ -= 1
                while e_ + 1 < len(act) and act[e_ + 1]:
                    e_ += 1
                for i in range(s_, e_ + 1):
                    fin[i] = 1
            p = os.path.join(S.EVIDENCE, cat, a["evidence"])
            img = cv2.imread(p)
            if img is None:
                continue
            small = cv2.resize(img, S.VQA_SIZE, interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
            items.append({
                "kind": kind, "cat": cat, "cat_dir": c["cat_dir"], "clip": c["name"],
                "sec": a["sec"], "label": lab, "gt_at": int(gt[a["idx"]]),
                "gt_span": "".join(map(str, gt)),
                "act": "".join(map(str, act)),
                "fin": "".join(map(str, fin)),
                "solo": "".join(map(str, solo.get((c["cat_dir"], c["name"], cat), []))),
                "res": f"{w}x{h}",
                "big": b64(cv2.resize(img, (560, int(560 * h / w)))),
                "small": b64(cv2.resize(small, (S.VQA_SIZE[0] * 2, S.VQA_SIZE[1] * 2),
                                        interpolation=cv2.INTER_NEAREST)),
            })

    order = ["정탐 기각", "오탐 통과", "정탐 통과", "오탐 기각"]
    items.sort(key=lambda x: (order.index(x["kind"]), x["cat"], x["clip"]))
    cnt = {k: sum(1 for i in items if i["kind"] == k) for k in order}

    html = TPL.replace("__DATA__", json.dumps(items, ensure_ascii=False))
    html = html.replace("__COUNT__", json.dumps(cnt, ensure_ascii=False))
    path = os.path.join(OUT, "index.html")
    open(path, "w", encoding="utf-8").write(html)
    print(f"알람 {len(items)}건  " + "  ".join(f"{k} {v}" for k, v in cnt.items()))
    print(f"-> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


TPL = r"""<!doctype html><meta charset="utf-8"><title>2차 판정 검토</title>
<style>
 body{margin:0;font:14px/1.5 system-ui,"Noto Sans KR",sans-serif;background:#111;color:#eee}
 header{position:sticky;top:0;background:#1b1b1b;border-bottom:1px solid #333;padding:10px 16px;z-index:9}
 h1{margin:0 0 8px;font-size:16px}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:5px;
        padding:6px 12px;margin-right:6px;cursor:pointer;font:inherit}
 button.on{background:#2563eb;border-color:#2563eb;color:#fff}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:14px;padding:14px}
 .card{background:#1b1b1b;border:1px solid #333;border-radius:8px;overflow:hidden}
 .card.정탐기각{border-color:#dc2626;border-width:2px}
 .card.오탐통과{border-color:#d97706}
 .hd{padding:7px 10px;font-size:12px;display:flex;justify-content:space-between;gap:8px}
 .tag{padding:1px 8px;border-radius:10px;font-weight:700;white-space:nowrap}
 .정탐기각 .tag{background:#dc2626}.오탐통과 .tag{background:#d97706}
 .정탐통과 .tag{background:#16a34a}.오탐기각 .tag{background:#475569}
 .ims{display:flex;gap:6px;padding:0 8px 6px;align-items:flex-start}
 .ims img{display:block;border:1px solid #444}
 .ims .l{flex:1}.ims .l img{width:100%}
 .cap{font-size:10px;color:#888;text-align:center;padding-top:2px}
 .mono{font-family:ui-monospace,monospace;font-size:11px;color:#9ca3af;padding:0 10px 9px;
       word-break:break-all}
 .mono b{color:#e5e7eb}
 .gtl{letter-spacing:2px}
 .g1{color:#22c55e}.a1{color:#60a5fa}.f1c{color:#f59e0b}.s1c{color:#e879f9}
</style>
<header>
 <h1>2차 검출기 판정 검토 — 알람 <span id="n"></span>건</h1>
 <div id="btns"></div>
 <div style="font-size:11px;color:#888;margin-top:6px">
  왼쪽 = 원본 · 오른쪽 = <b>2차가 실제로 본 256×192</b>(2배 확대).
  막대는 1초 격자. <span class="g1">초록=GT</span> ·
  <span class="a1">파랑=<b>1차 알람만</b>(2차 무관)</span> ·
  <span class="f1c">주황=2단계 최종(2차 통과분)</span> ·
  <span class="s1c">보라=2차 단독(1차·FSM 없이 매 초 VQA)</span>.
  2차가 기각하면 1차가 아무리 길게 울렸어도 최종은 전부 0 이 된다.
 </div>
</header>
<div class="grid" id="g"></div>
<script>
const D=__DATA__, C=__COUNT__;
const ORDER=["정탐 기각","오탐 통과","정탐 통과","오탐 기각"];
let cur="정탐 기각";
document.getElementById('n').textContent=D.length;
const bw=document.getElementById('btns');
["전체"].concat(ORDER).forEach(k=>{
  const b=document.createElement('button');
  b.textContent = k==="전체" ? `전체 ${D.length}` : `${k} ${C[k]||0}`;
  b.onclick=()=>{cur=k;draw();[...bw.children].forEach(x=>x.classList.remove('on'));b.classList.add('on')};
  bw.appendChild(b); if(k===cur) b.classList.add('on');
});
function bar(s,cls){return [...s].map(c=>c==='1'?`<span class="${cls}">1</span>`:'0').join('')}
function draw(){
  const g=document.getElementById('g');
  const rows=D.filter(d=>cur==="전체"||d.kind===cur);
  g.innerHTML=rows.map(d=>`
   <div class="card ${d.kind.replace(' ','')}">
    <div class="hd"><span><span class="tag">${d.kind}</span>
      &nbsp;<b>${d.cat}</b> · 2차 <b>${d.label}</b></span>
      <span style="color:#777">${d.sec.toFixed(1)}s · ${d.res}</span></div>
    <div class="ims">
      <div class="l"><img src="data:image/jpeg;base64,${d.big}"><div class="cap">원본</div></div>
      <div><img src="data:image/jpeg;base64,${d.small}"><div class="cap">2차 입력 256×192</div></div>
    </div>
    <div class="mono">${d.clip} <span style="color:#666">(${d.cat_dir} 폴더)</span><br>
      <b>GT</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="gtl">${bar(d.gt_span,'g1')}</span>
       &nbsp; 알람 시점 GT=<b>${d.gt_at}</b><br>
      <b>1차</b>&nbsp;&nbsp;&nbsp;<span class="gtl">${bar(d.act,'a1')}</span>
       &nbsp; <span style="color:#777">2차 입력은 발화 시점 1장뿐</span><br>
      <b>최종</b>&nbsp;<span class="gtl">${bar(d.fin,'f1c')}</span>
       &nbsp; <span style="color:#777">2차 통과분만 — 파이프라인 F1 은 이 줄</span>
      ${d.solo ? `<br><b>2차단독</b> <span class="gtl">${bar(d.solo,'s1c')}</span>
       &nbsp; <span style="color:#777">1차·FSM 없이 매 초 VQA</span>` : ''}</div>
   </div>`).join('') || '<p style="padding:20px;color:#888">해당 없음</p>';
}
draw();
</script>
"""

if __name__ == "__main__":
    main()
