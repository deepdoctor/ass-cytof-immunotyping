"""
Patch 04_umap_celltype.png to de-emphasise the two artifact populations
(T-Myeloid doublets, CD16+ Granulocyte) which are excluded from all
downstream analyses but currently appear in bright red text on the UMAP
and in the legend, suggesting visual parity with the real cell types.

Two interventions:
  (1) Overlay a grey footnote box at the bottom-left explaining that
      these two populations are excluded artifacts (so a reader who only
      glances at the figure understands the status without consulting
      methods).
  (2) Mask + re-render the two on-UMAP red text labels in grey italic
      with an '(excluded)' suffix.

Without the AnnData h5ad we cannot recolour the underlying point cloud,
but the footnote + recoloured labels make the artifact status legible.
"""
import os, shutil
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "./cytof_output_nature"
SRC = f"{OUT_DIR}/04_umap_celltype.png"
BAK = f"{OUT_DIR}/04_umap_celltype.original.png"

if not os.path.exists(BAK):
    shutil.copy(SRC, BAK)
    print(f"Backed up: {BAK}")

img = Image.open(SRC).convert("RGB")
W, H = img.size
print(f"Image: {W}×{H}")
draw = ImageDraw.Draw(img)

# Font choices — Arial Italic for "(excluded)" hint, regular for footnote.
def load_font(size, italic=False, bold=False):
    candidates = [
        # Italic variants
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf"     if italic else None,
        "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf" if italic and bold else None,
        # Bold
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"        if bold and not italic else None,
        # Regular fallback
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for fp in candidates:
        if fp and os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()

# ─── (1) on-UMAP artifact-label patches ────────────────────────────────────
# Approximate centres of the two red labels in the existing render.
# Tuned by visual inspection of the 1321×1170 source PNG.
ARTIFACT_LABELS = [
    {
        # 'T-Myeloid doublets'
        "frac_x":  0.158,
        "frac_y":  0.052,
        "half_w_frac": 0.094,
        "half_h_frac": 0.018,
        "new_text": "T-Myeloid doublets (excluded)",
    },
    {
        # 'CD16+ Granulocyte'
        "frac_x":  0.183,
        "frac_y":  0.235,
        "half_w_frac": 0.085,
        "half_h_frac": 0.018,
        "new_text": "CD16+ Granulocyte (excluded)",
    },
]

label_font = load_font(size=max(14, int(H * 0.018)), italic=True)
GREY = "#7F7F7F"

for cfg in ARTIFACT_LABELS:
    cx = cfg["frac_x"] * W
    cy = cfg["frac_y"] * H
    hw = cfg["half_w_frac"] * W
    hh = cfg["half_h_frac"] * H
    # mask
    draw.rectangle([int(cx - hw), int(cy - hh),
                    int(cx + hw), int(cy + hh)], fill="white")
    # write replacement
    txt = cfg["new_text"]
    bbox = draw.textbbox((0, 0), txt, font=label_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(cx - tw / 2)
    ty = int(cy - th / 2)
    draw.text((tx, ty), txt, fill=GREY, font=label_font)

# ─── (2) footnote annotation ───────────────────────────────────────────────
# Place a small grey box at the bottom-left of the image, well clear of the
# legend (which lives at bottom-right in the source render).
fn_font = load_font(size=max(11, int(H * 0.014)))
FOOTNOTE = (
    "* T-Myeloid doublets and CD16+ Granulocyte: "
    "technical artifacts excluded from all downstream analyses (see Methods §M3)."
)
fn_box_x0 = int(W * 0.012)
fn_box_y0 = int(H * 0.962)
# Estimate text box size
bbox = draw.textbbox((0, 0), FOOTNOTE, font=fn_font)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
pad = 6
draw.rectangle(
    [fn_box_x0 - pad, fn_box_y0 - pad,
     fn_box_x0 + tw + pad, fn_box_y0 + th + pad],
    fill="#F5F5F5", outline="#BBB", width=1,
)
draw.text((fn_box_x0, fn_box_y0), FOOTNOTE, fill="#444", font=fn_font)

img.save(SRC, dpi=(300, 300))
print(f"Patched: {SRC}")
print(f"  Artifact labels recoloured grey + '(excluded)' suffix.")
print(f"  Footnote added at bottom-left.")
