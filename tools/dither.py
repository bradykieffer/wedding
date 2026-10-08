# /// script
# dependencies = ["numpy", "pillow"]
# ///
# Usage: uv run tools/dither.py SRC OUT WIDTH HEXPALETTE atkinson|fs GAMMA L,T,R,B
# e.g.   uv run tools/dither.py photo.jpg photo-440.png 440 f2eee5,d4201b,141414 atkinson 0.5 0.07,0.12,0.93,0.98
# Dither a photo down to the page's inks. Atkinson spreads only 3/4 of the error,
# so it's crisper but eats shadows; Floyd-Steinberg keeps midtones.
import sys
import numpy as np
from PIL import Image, ImageOps

src, out, width, palette, algo, gamma = sys.argv[1:7]
crop = [float(v) for v in sys.argv[7].split(",")]  # left, top, right, bottom as fractions
width, gamma = int(width), float(gamma)
pal = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in palette.split(",")], float)
taps = {
    "atkinson": [(0, 1, 1 / 8), (0, 2, 1 / 8), (1, -1, 1 / 8), (1, 0, 1 / 8), (1, 1, 1 / 8), (2, 0, 1 / 8)],
    "fs": [(0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)],
}[algo]

im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
W, H = im.size
im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
im = ImageOps.autocontrast(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS), cutoff=1)
a = 255 * (np.asarray(im, float) / 255) ** gamma
h, w, _ = a.shape
idx = np.zeros((h, w), np.uint8)
for y in range(h):
    for x in range(w):
        px = a[y, x]
        k = int(((pal - px) ** 2).sum(1).argmin())
        idx[y, x] = k
        err = px - pal[k]
        for dy, dx, f in taps:
            if y + dy < h and 0 <= x + dx < w:
                a[y + dy, x + dx] += err * f
img = Image.fromarray(idx, "P")
img.putpalette(pal.astype(np.uint8).flatten().tolist())
img.save(out, optimize=True)
