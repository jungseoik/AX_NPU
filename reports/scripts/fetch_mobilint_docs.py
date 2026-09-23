"""docs.mobilint.com(Sphinx) 문서를 마크다운으로 받아 docs/ 에 저장.

사이트의 "Download source file" 버튼이 가리키는 **`_sources/<page>.md` 원본을 그대로** 받는다.
원본이 없는 페이지만 HTML 을 변환해 보완한다(bs4 사용, 외부 변환기 불필요).

    python reports/scripts/fetch_mobilint_docs.py --set compiler --version v1.3
    python reports/scripts/fetch_mobilint_docs.py --set runtime  --version v1.4 --lang kr
"""
from __future__ import annotations
import argparse, re, sys, time, urllib.request
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString

BASE = "https://docs.mobilint.com/{s}/{v}/{lang}/"


def get(url: str) -> str:
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == 2:
                raise
            time.sleep(2)


def page_list(base: str) -> list[str]:
    soup = BeautifulSoup(get(base + "introduction.html"), "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        h = a["href"].split("#")[0]
        if (h.endswith(".html") and "/" not in h.strip("./")
                and not h.startswith(("http", "../"))
                and h not in ("genindex.html", "search.html")):
            out.append(h.lstrip("./"))
    return sorted(set(out))


def md_of(node, depth=0) -> str:
    """Sphinx 본문 노드를 마크다운으로. 표/코드/목록/헤딩만 다룬다."""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    name = node.name
    if name in ("script", "style"):
        return ""
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        t = node.get_text(" ", strip=True).replace("¶", "").strip()
        return f"\n\n{'#' * int(name[1])} {t}\n"
    if name in ("pre",):
        code = node.get_text("", strip=False).rstrip()
        lang = ""
        cls = " ".join(node.get("class", [])) + " " + " ".join(
            node.find("code").get("class", []) if node.find("code") else [])
        for l in ("python", "bash", "shell", "cpp", "c++", "json", "yaml"):
            if l in cls.lower():
                lang = "python" if l == "python" else ("bash" if l in ("bash", "shell") else l)
                break
        return f"\n\n```{lang}\n{code}\n```\n"
    if name == "code":
        return f"`{node.get_text('', strip=True)}`"
    if name in ("strong", "b"):
        return f"**{node.get_text(' ', strip=True)}**"
    if name in ("em", "i"):
        return f"*{node.get_text(' ', strip=True)}*"
    if name == "a" and node.get("href"):
        t = node.get_text(" ", strip=True)
        h = node["href"]
        return f"[{t}]({h})" if t else ""
    if name == "table":
        rows = []
        for tr in node.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * n) + " |"]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n\n" + "\n".join(out) + "\n"
    if name in ("ul", "ol"):
        items = []
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            body = "".join(md_of(c, depth + 1) for c in li.children).strip()
            body = re.sub(r"\n{2,}", "\n", body)
            mark = "-" if name == "ul" else f"{i}."
            pad = "  " * depth
            items.append(f"{pad}{mark} {body}")
        return "\n\n" + "\n".join(items) + "\n"
    if name in ("p", "div", "section", "dl", "dd", "dt", "li", "span", "blockquote", "figure"):
        inner = "".join(md_of(c, depth) for c in node.children)
        return ("\n\n" + inner.strip() + "\n") if name in ("p", "blockquote") else inner
    return "".join(md_of(c, depth) for c in node.children)


def convert(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = (soup.find("div", {"role": "main"}) or soup.find("main")
            or soup.find("div", class_=re.compile("body|document")))
    if main is None:
        return ""
    for sel in ("nav", "footer", "header"):
        for t in main.find_all(sel):
            t.decompose()
    for t in main.find_all("a", class_="headerlink"):
        t.decompose()
    md = md_of(main)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return f"<!-- 출처: {url} (자동 변환) -->\n\n{md}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["compiler", "runtime", "driver"])
    ap.add_argument("--version", required=True, help="예: v1.3 / v1.4")
    ap.add_argument("--lang", default="en", choices=["en", "kr"])
    ap.add_argument("--out", default=None, help="기본: docs/<set>_<version>")
    a = ap.parse_args()

    base = BASE.format(s=a.set, v=a.version, lang=a.lang)
    out = Path(a.out or f"docs/{a.set}_{a.version}")
    out.mkdir(parents=True, exist_ok=True)
    pages = page_list(base)
    print(f"[plan] {base}  페이지 {len(pages)}개 → {out}", flush=True)
    ok = 0
    for p in pages:
        stem = p[:-5]
        try:
            md, src = None, None
            # ① 원본 마크다운 우선. 사이트마다 _sources 위치가 다르다.
            #    compiler: <lang>/_sources/<page>.md   runtime: ../_sources/<lang>/<page>.md
            for cand in (f"{base}_sources/{stem}.md",
                         f"{base}../_sources/{a.lang}/{stem}.md"):
                try:
                    md, src = get(cand), "원본"
                    break
                except Exception:
                    continue
            if md is None:                         # ② 없으면 HTML 변환
                md, src = convert(get(base + p), base + p), "HTML변환"
            if not md.strip():
                print(f"  [skip] {p} (본문 없음)"); continue
            head = f"<!-- 출처: {base}{p} ({src}) -->\n\n"
            (out / f"{stem}.md").write_text(head + md.lstrip(), encoding="utf-8")
            print(f"  [ok] {stem+'.md':34s} {len(md):6d}자  ({src})", flush=True); ok += 1
        except Exception as e:
            print(f"  [FAIL] {p}: {type(e).__name__}: {e}", flush=True)
    print(f"\n[done] {ok}/{len(pages)} → {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
