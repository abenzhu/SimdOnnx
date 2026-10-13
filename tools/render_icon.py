"""Renders assets/icon.png (1024/256/128) from the same geometry as assets/icon.svg."""
from pathlib import Path
from PIL import Image, ImageDraw

DARK, LIGHT = (31, 53, 224, 255), (140, 144, 245, 255)
S = 4                      # supersampling
N = 1024 * S
C = N // 2

def p(v): return int(round(v * S))

def render() -> Image.Image:
    img = Image.new('RGBA', (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = 512
    # pins (drawn first; the chip stroke covers their inner ends)
    for y in (-165, -55, 55, 165):
        for x0, x1 in ((-392, -280), (280, 392)):
            d.rounded_rectangle([p(cx + x0), p(cy + y - 20), p(cx + x1), p(cy + y + 20)], radius=p(20), fill=LIGHT)
    # chip body
    d.rounded_rectangle([p(cx - 300), p(cy - 300), p(cx + 300), p(cy + 300)], radius=p(96), fill=(255, 255, 255, 255),
                        outline=DARK, width=p(56))
    # compute graph: 1 -> 2 -> 1
    nodes = {'a': (-170, 0), 'b': (-55, -110), 'c': (-55, 110), 'd': (60, 0)}
    for u, v in (('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')):
        (x0, y0), (x1, y1) = nodes[u], nodes[v]
        d.line([p(cx + x0), p(cy + y0), p(cx + x1), p(cy + y1)], fill=LIGHT, width=p(26))
    for x, y in nodes.values():
        d.ellipse([p(cx + x - 40), p(cy + y - 40), p(cx + x + 40), p(cy + y + 40)], fill=DARK)
    # SIMD lanes
    for y in (-135, -45, 45, 135):
        d.rounded_rectangle([p(cx + 135), p(cy + y - 24), p(cx + 225), p(cy + y + 24)], radius=p(24), fill=DARK)
    return img.rotate(10, resample=Image.BICUBIC)

if __name__ == '__main__':
    out = Path(__file__).resolve().parent.parent / 'assets'
    out.mkdir(exist_ok=True)
    big = render()
    for size, name in ((1024, 'icon.png'), (256, 'icon-256.png'), (128, 'icon-128.png')):
        big.resize((size, size), Image.LANCZOS).save(out / name)
        print(out / name)
