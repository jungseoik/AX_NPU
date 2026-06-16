"""
튜토리얼용 예제 이미지 다운로드.

공개 COCO val2017 이미지 몇 장을 직접 URL로 받는다(로그인 불필요).
서로 다른 카테고리(고양이/개/버스/피자 등) + 비슷한 쌍을 섞어, 추론 후 임베딩
유사도가 '비슷한 이미지는 높고 다른 이미지는 낮은지' 확인할 수 있게 한다.

사용:
    python download_images.py            # ./images 에 저장
    python download_images.py --out ./images
"""
import argparse
import os
import urllib.request

# COCO val2017 공개 이미지 (images.cocodataset.org, 로그인 불필요)
URLS = {
    "cat1.jpg":  "http://images.cocodataset.org/val2017/000000039769.jpg",  # 고양이 두 마리
    "cat2.jpg":  "http://images.cocodataset.org/val2017/000000000139.jpg",  # 실내(고양이/TV)
    "dog.jpg":   "http://images.cocodataset.org/val2017/000000000785.jpg",  # 스키/사람
    "bus.jpg":   "http://images.cocodataset.org/val2017/000000000632.jpg",  # 실내 가구
    "pizza.jpg": "http://images.cocodataset.org/val2017/000000001000.jpg",  # 음식류
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, url in URLS.items():
        dst = os.path.join(args.out, name)
        if os.path.exists(dst):
            print(f"  이미 있음: {name}")
            continue
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  받음: {name}  ({os.path.getsize(dst)//1024} KB)  <- {url}")
        except Exception as e:
            print(f"  실패: {name}  ({e})")

    n = len([f for f in os.listdir(args.out) if f.lower().endswith((".jpg", ".png"))])
    print(f"[done] {n}장 → {args.out}")


if __name__ == "__main__":
    main()
