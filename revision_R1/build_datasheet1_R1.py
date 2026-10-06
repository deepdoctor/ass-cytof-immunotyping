"""
Revision R1 -- rebuild Supplementary Data Sheet 1 with corrected Figs. S9, S10 and
S12, a redrawn S8 and the corrected S13 legend and labels.  Every other page is copied unchanged; the three pages are re-laid out with
the original geometry (28-pt margins, Arial Unicode MS 9/7.5 pt, 10.9-pt leading).
Usage: python revision_R1/build_datasheet1_R1.py "<Data Sheet 1.PDF>" <out.pdf>
"""
import sys
import fitz

SRC, DST = sys.argv[1], sys.argv[2]
FONT = "/Library/Fonts/Arial Unicode.ttf"
FIG = "revision_R1/figs_R1/"
REPLACE = {
    8: (FIG + "SuppFig_S8_validation_revised.pdf",
        [("18-sample draws (p = 0.19)", "18-sample draws (p = 0.21)")]),
    9: (FIG + "SuppFig_S9_IFN_revised.pdf",
        [("STAT2 — 11/13 detected", "STAT2 — 13/13 detected"),
         ("Kirou et al. 2005 — 3/4 detected", "Kirou et al. 2005 — 4/4 detected"),
         ("AUC for ASyS-vs-DM 0.20 and 0.23", "AUC for ASyS-vs-DM 0.21 and 0.23")]),
    10: (FIG + "SuppFig_S10_withinASyS_revised.pdf",
         [("3,266 genes at FDR", "4,327 genes at FDR")]),
    12: (FIG + "SuppFig_S12_JDM_revised.pdf", []),
    13: (None, [("from I²=62% to I²=17%", "from I²=44% to I²=17%")]),   # figure unchanged
}
M, GAP_T, GAP_B, LEAD, BOT = 28.0, 15.0, 17.5, 10.9, 21.6
font = fitz.Font(fontfile=FONT)


def wrap(text, size, width):
    lines, cur = [], ""
    for w in text.split(" "):
        trial = (cur + " " + w).strip()
        if font.text_length(trial, fontsize=size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


src = fitz.open(SRC)
# S13 (page 13): relabel the two "AS" rows of panel c as "ASyS", right edge kept
p13 = src[12]
fixes = []
for b in p13.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        for sp in l["spans"]:
            if sp["text"].startswith("AS  "):
                fixes.append((fitz.Rect(sp["bbox"]), sp["origin"], sp["size"], "ASyS" + sp["text"][2:]))
assert len(fixes) == 2, fixes
for r, _, _, _ in fixes:
    p13.add_redact_annot(fitz.Rect(r.x0, r.y0 + 1.0, r.x1, r.y1 - 1.0))
p13.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
for r, org, size, txt in fixes:
    w = fitz.get_text_length(txt, fontname="helv", fontsize=size)
    p13.insert_text((r.x1 - w, org[1]), txt, fontname="helv", fontsize=size)
out = fitz.open()
for i, pg in enumerate(src):
    n = i + 1
    if n not in REPLACE:
        out.insert_pdf(src, from_page=i, to_page=i)
        continue
    figfile, subs = REPLACE[n]
    t = pg.search_for(f"Supplementary Figure S{n} |")[0]
    lines = [l for b in pg.get_text("dict")["blocks"] for l in b.get("lines", [])
             if l["bbox"][1] >= t.y0 - 1 and "ArialUnicode" in l["spans"][0]["font"]]
    title = "".join(s["text"] for s in lines[0]["spans"])
    body = " ".join("".join(s["text"] for s in l["spans"]) for l in lines[1:])
    for old, new in subs:
        assert body.count(old) == 1, (n, old)
        body = body.replace(old, new)
    W = pg.rect.width
    fw = W - 2 * M
    if figfile is None:                       # keep the original figure
        clip = fitz.Rect(M, 20.0, W - M, t.y0 - GAP_T)
        fig, fh = src, clip.height
    else:
        clip = None
        fig = fitz.open(figfile)
        fh = fw * fig[0].rect.height / fig[0].rect.width
    tl = wrap(title, 9.0, fw); bl = wrap(body, 7.5, fw)
    y_title = 20.0 + fh + GAP_T
    y_body = y_title + GAP_B + (len(tl) - 1) * 12.0
    H = y_body + (len(bl) - 1) * LEAD + 10.04 + BOT
    np_ = out.new_page(width=W, height=H)
    if clip is None:
        np_.show_pdf_page(fitz.Rect(M, 20.0, M + fw, 20.0 + fh), fig, 0)
    else:
        np_.show_pdf_page(fitz.Rect(M, 20.0, M + fw, 20.0 + fh), src, i, clip=clip)
    np_.insert_font(fontname="aum", fontfile=FONT)
    asc = font.ascender
    for k, s in enumerate(tl):
        np_.insert_text((M, y_title + k * 12.0 + asc * 9.0), s, fontname="aum", fontsize=9.0)
    for k, s in enumerate(bl):
        np_.insert_text((M, y_body + k * LEAD + asc * 7.5), s, fontname="aum", fontsize=7.5)
out.subset_fonts()
out.save(DST, garbage=4, deflate=True, clean=True)
print("saved", DST, len(out), "pages")
