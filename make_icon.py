"""홈 화면 아이콘 생성.

iOS 는 apple-touch-icon 이 없으면 페이지를 축소한 스크린샷을 아이콘으로 쓴다.
리포트 팔레트 그대로, 이동평균선(점선)을 주가선이 뚫고 올라가는 모티프.

  python make_icon.py

4배 크기로 그린 뒤 축소해 계단현상을 없앤다. iOS 가 알아서 둥근 사각형으로
잘라내므로 여백만 남기고 배경은 꽉 채운다.
"""
from PIL import Image, ImageDraw

S = 180          # 최종 크기 (iPhone 홈 화면 기준)
F = 4            # 안티에일리어싱용 배율
W = S * F

GROUND = (20, 27, 30)      # --ink 계열 짙은 슬레이트
PRICE = (242, 245, 245)    # 주가선 (밝게)
ACCENT = (227, 162, 78)    # --accent 앰버, 이동평균선

# 720 좌표계 기준 구성 — iOS 둥근 모서리에 잘리지 않도록 안쪽에 배치
MA = [(105, 335), (615, 300)]
PRICE_PTS = [(105, 505), (250, 470), (385, 495), (505, 295), (615, 195)]


def at(pts, x):
    """폴리라인 위 x 지점의 y."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def crossing():
    """주가선이 이동평균선을 위로 뚫는 지점."""
    lo, hi = PRICE_PTS[0][0], PRICE_PTS[-1][0]
    for _ in range(60):
        mid = (lo + hi) / 2
        if at(PRICE_PTS, mid) > at(MA, mid):   # y 가 크면 아래
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def draw(size):
    img = Image.new("RGB", (W, W), GROUND)
    d = ImageDraw.Draw(img)

    # 이동평균선 — 점선. 주가선보다 먼저 그려 아래에 깔린다.
    (x0, y0), (x1, y1) = MA
    dash, gap, x = 46, 30, x0
    while x < x1:
        xe = min(x + dash, x1)
        d.line([(x, at(MA, x)), (xe, at(MA, xe))], fill=ACCENT, width=28)
        x = xe + gap

    # 주가선 — 점선 위를 덮으며 지나간다 (선을 끊지 않는다)
    d.line(PRICE_PTS, fill=PRICE, width=48, joint="curve")
    for pt in PRICE_PTS[1:-1]:                     # 꺾이는 지점 둥글게
        d.ellipse([pt[0] - 24, pt[1] - 24, pt[0] + 24, pt[1] + 24], fill=PRICE)

    # 현재가 — 선 끝의 앰버 점
    ex, ey = PRICE_PTS[-1]
    d.ellipse([ex - 38, ey - 38, ex + 38, ey + 38], fill=ACCENT)

    return img.resize((size, size), Image.LANCZOS)


for size, name in ((180, "apple-touch-icon.png"), (512, "icon-512.png"), (32, "favicon.png")):
    draw(size).save(name)
    print(f"  {name} ({size}x{size})")
