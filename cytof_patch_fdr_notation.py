"""
Patch the FDR notation in source PNGs that were generated before the
fmt_p() fix to cytof_plots_nature.py.

Targets the two violin grids whose sub-panel titles use the unprofessional
'FDR = 0.0e+00' notation produced when KW p-values underflow below 1e-308:

  - 07_top_marker_violins.png         (Fig 2c source)  9 panels, 3×3
  - 08b_within_celltype_violins.png   (Fig 5b source)  9 panels, 3×3

Replaces each panel title with a consistent ASCII-safe rendering:
  'MARKER  P < 10^-300' (for underflow) or 'MARKER  P = 9.5e-157' (for valid).

After running this, re-run cytof_composite_figures.py to refresh the
MainFig2 and MainFig5 composites that embed these PNGs.
"""
import os, shutil
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "./cytof_output_nature"

CANDIDATE_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]

def load_font(H, frac=0.026, lo=20):
    for fp in CANDIDATE_FONTS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=max(lo, int(H * frac)))
            except Exception:
                continue
    return ImageFont.load_default()


def fmt_padj(s):
    """ '0.0e+00' → 'P < 10^-300'; valid sci → 'P = X.Xe-YYY' """
    s = s.strip()
    if s in {"0.0e+00", "0.00e+00", "0e+00"}:
        return "P < 10^-300"
    return f"P = {s}"


def patch_grid_3x3(src_path, titles, row_y_fracs, col_x_fracs,
                   band_half_h_frac=0.030, band_half_w_frac=0.165,
                   font_frac=0.026, multi_line=False):
    """Patch a 3×3 panel-title grid in place. Backs up to *.original.png.

    titles: list of either str (single-line) or (top, bottom) tuples.
    """
    bak = src_path.replace(".png", ".original.png")
    if not os.path.exists(bak):
        shutil.copy(src_path, bak)
        print(f"  Backed up: {bak}")

    img = Image.open(src_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = load_font(H, frac=font_frac)

    band_half_h = max(14, int(H * band_half_h_frac))
    band_half_w = int(W * band_half_w_frac)

    for r in range(3):
        for c in range(3):
            entry = titles[r * 3 + c]
            cx = col_x_fracs[c] * W
            cy = row_y_fracs[r] * H

            # mask
            x0, x1 = int(cx - band_half_w), int(cx + band_half_w)
            y0, y1 = int(cy - band_half_h), int(cy + band_half_h)
            draw.rectangle([x0, y0, x1, y1], fill="white")

            if multi_line:
                top, bot = entry
                for line, dy in [(top, -band_half_h * 0.45),
                                 (bot,  band_half_h * 0.45)]:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    tx = int(cx - tw / 2)
                    ty = int(cy + dy - th / 2)
                    draw.text((tx, ty), line, fill="#222222", font=font)
            else:
                line = entry
                bbox = draw.textbbox((0, 0), line, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = int(cx - tw / 2)
                ty = int(cy - th / 2)
                draw.text((tx, ty), line, fill="#222222", font=font)

    img.save(src_path, dpi=(300, 300))
    print(f"  Patched: {src_path}")


# ── 07_top_marker_violins.png ───────────────────────────────────────────────
print("[1/2] 07_top_marker_violins.png")
TOP9_GLOBAL = ["CD45", "CD4", "HLA-DR", "CD24", "CCR6", "GranzymeB",
               "CD19", "CD127", "IgM"]
titles_07 = [f"{m}    P < 10^-300" for m in TOP9_GLOBAL]
patch_grid_3x3(
    src_path=f"{OUT_DIR}/07_top_marker_violins.png",
    titles=titles_07,
    row_y_fracs=[0.150, 0.450, 0.745],
    col_x_fracs=[0.200, 0.515, 0.835],
    band_half_h_frac=0.030,
    band_half_w_frac=0.165,
    font_frac=0.026,
    multi_line=False,
)

# ── 08b_within_celltype_violins.png ─────────────────────────────────────────
print("[2/2] 08b_within_celltype_violins.png")
WITHIN_CT_TOP9 = [
    ("CD4 Naive T",          "CD45RB",     "0.0e+00"),
    ("CD4 Central Memory T", "CD45RO",     "0.0e+00"),
    ("CD4 Effector Memory T","CD4",        "9.5e-157"),
    ("CD4 Th1-like",         "GranzymeB",  "5.8e-272"),
    ("CD8 Effector T",       "CD38",       "0.0e+00"),
    ("CD8 Naive T",          "CD27",       "6.8e-100"),
    ("NK cell",              "CD38",       "5.2e-256"),
    ("Naive B",              "IgD",        "2.3e-197"),
    ("Classical Monocyte",   "CD14",       "0.0e+00"),
]
titles_08b = [(ct, f"{mk}   {fmt_padj(p)}") for (ct, mk, p) in WITHIN_CT_TOP9]
patch_grid_3x3(
    src_path=f"{OUT_DIR}/08b_within_celltype_violins.png",
    titles=titles_08b,
    row_y_fracs=[0.150, 0.460, 0.760],
    col_x_fracs=[0.200, 0.515, 0.835],
    band_half_h_frac=0.055,
    band_half_w_frac=0.170,
    font_frac=0.022,
    multi_line=True,
)

print("Done.")
