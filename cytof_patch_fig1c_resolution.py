"""
Patch the title of 02_umap_clusters.png from 'resolution = 0.5' (rendering bug)
to 'resolution = 0.8' (correct value, matches Methods §M4 and main text).

The h5ad source for full re-rendering is not in the repo; this is a clean
in-place title patch using PIL. The new MainFig1_atlas composite then uses
the patched PNG via the existing cytof_composite_figures.py pipeline.
"""
from PIL import Image, ImageDraw, ImageFont
import os, shutil

OUT_DIR = "./cytof_output_nature"
SRC = f"{OUT_DIR}/02_umap_clusters.png"
BAK = f"{OUT_DIR}/02_umap_clusters.original.png"

if not os.path.exists(BAK):
    shutil.copy(SRC, BAK)
    print(f"Backed up original to {BAK}")

img = Image.open(SRC).convert("RGB")
W, H = img.size
print(f"Image size: {W} × {H}")

# The title sits in a horizontal band near the top of the panel.
# We white out a horizontal band that covers the title text vertical extent,
# then draw the corrected title centred horizontally.
draw = ImageDraw.Draw(img)

# Mask region: top ~5% of image (typical matplotlib title area).
band_top = int(H * 0.005)
band_bot = int(H * 0.075)
# leave a margin from edges so we don't paint over panel-label "c" area
band_left  = int(W * 0.10)
band_right = int(W * 0.92)
draw.rectangle([band_left, band_top, band_right, band_bot], fill="white")

# Pick a good font — Arial is the matplotlib default for nature_style.
candidate_fonts = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
font = None
for fp in candidate_fonts:
    if os.path.exists(fp):
        try:
            # font size proportional to image height; matches matplotlib 26pt
            # at 300 dpi (~108 px per inch -> 26pt -> ~108 px). Scale to image:
            font = ImageFont.truetype(fp, size=max(14, int(H * 0.030)))
            break
        except Exception:
            continue
if font is None:
    font = ImageFont.load_default()
    print("WARNING: using default font; size may not match")

new_title = "Leiden clusters  (resolution = 0.8)"
# Pillow ≥ 8: textbbox gives accurate width
try:
    bbox = draw.textbbox((0, 0), new_title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
except AttributeError:
    tw, th = draw.textsize(new_title, font=font)

x = (W - tw) // 2
y = (band_top + band_bot) // 2 - th // 2
draw.text((x, y), new_title, fill="#222222", font=font)

img.save(SRC, dpi=(300, 300))
print(f"Patched: {SRC}")
print(f"  '… resolution = 0.5' → '… resolution = 0.8'")
