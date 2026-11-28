"""Minimal SVG drawing helpers for the GPT-2 paper-figure illustrations (copied from the GPT-1 set).

Style: paper-figure — white background, dark-gray ink, two accents.
Colour semantics (fixed for the whole set, see paper-comic SKILL.md Step 4.1):
  BLUE   = 语言模型 / GPT-2（模型本体、它的预测与结果）
  ORANGE = 任务提示（写成文字的任务：TL;DR:、A:、english = french ……）
  grey   = 数据、输入、baseline / 此前 SOTA
  INK    = 线条与文字
Never use BLUE or ORANGE for anything else (not for "important", not for "beat SOTA").
Rendering: headless Chrome (local, no network).
"""
import html
import math
import os
import re
import shutil
import subprocess
import sys

W, H = 1600, 900

INK = "#1F2937"
MUTED = "#6B7280"
FAINT = "#9CA3AF"
LINE = "#D1D5DB"
PANEL = "#F8FAFC"
PANEL2 = "#F1F5F9"
BLUE = "#2563EB"
BLUE_L = "#DBEAFE"
BLUE_LL = "#EFF6FF"
ORANGE = "#EA580C"
ORANGE_L = "#FFEDD5"
ORANGE_LL = "#FFF7ED"

FONT = "Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
MATH = "DejaVu Serif, Times New Roman, serif"

TOTAL_PAGES = 6
SOURCE = "Radford et al., Language Models are Unsupervised Multitask Learners, OpenAI 2019"

# semantic aliases — use these, not raw colours
LM, LM_L, LM_LL = BLUE, BLUE_L, BLUE_LL
PROMPT, PROMPT_L, PROMPT_LL = ORANGE, ORANGE_L, ORANGE_LL
BASE, BASE_L = MUTED, PANEL2


def esc(s):
    return html.escape(str(s), quote=True)


_RICH = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|\*\*[^*]+\*\*)")


def rich(s, size):
    """Convert a mini markup to tspans: x_{k}, x^{T}, **bold**."""
    out = []
    small = round(size * 0.68, 1)
    for part in _RICH.split(s):
        if not part:
            continue
        if part.startswith("_{"):
            out.append(f'<tspan dy="{size*0.28:.1f}" font-size="{small}">{esc(part[2:-1])}</tspan>'
                       f'<tspan dy="{-size*0.28:.1f}">​</tspan>')
        elif part.startswith("^{"):
            out.append(f'<tspan dy="{-size*0.40:.1f}" font-size="{small}">{esc(part[2:-1])}</tspan>'
                       f'<tspan dy="{size*0.40:.1f}">​</tspan>')
        elif part.startswith("**"):
            out.append(f'<tspan font-weight="700">{esc(part[2:-2])}</tspan>')
        else:
            out.append(esc(part))
    return "".join(out)


def text(x, y, s, size=18, fill=INK, weight=400, anchor="start", family=FONT,
         italic=False, lh=1.35, rotate=None, opacity=None):
    lines = s if isinstance(s, (list, tuple)) else [s]
    attrs = (f'font-family="{family}" font-size="{size}" fill="{fill}" '
             f'font-weight="{weight}" text-anchor="{anchor}"')
    if italic:
        attrs += ' font-style="italic"'
    if opacity is not None:
        attrs += f' opacity="{opacity}"'
    if rotate is not None:
        attrs += f' transform="rotate({rotate} {x} {y})"'
    if len(lines) == 1:
        return f'<text x="{x}" y="{y}" {attrs}>{rich(lines[0], size)}</text>'
    spans = []
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else size * lh
        spans.append(f'<tspan x="{x}" dy="{dy:.1f}">{rich(ln, size)}</tspan>')
    return f'<text x="{x}" y="{y}" {attrs}>{"".join(spans)}</text>'


def ctext(x, y, s, size=18, **kw):
    """Centered text; y is the vertical center of the block."""
    lines = s if isinstance(s, (list, tuple)) else [s]
    lh = kw.get("lh", 1.35)
    total = size * lh * (len(lines) - 1)
    y0 = y - total / 2 + size * 0.36
    kw.setdefault("anchor", "middle")
    return text(x, round(y0, 1), lines, size, **kw)


def rect(x, y, w, h, fill="none", stroke=INK, sw=1.5, rx=10, dash=None, opacity=None):
    a = f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return f"<rect {a}/>"


def box(x, y, w, h, label=None, size=18, fill="#FFFFFF", stroke=INK, sw=1.6, rx=10,
        color=INK, weight=600, dash=None, sub=None, sub_size=14, sub_color=MUTED):
    parts = [rect(x, y, w, h, fill, stroke, sw, rx, dash)]
    if label is not None and sub is None:
        parts.append(ctext(x + w / 2, y + h / 2, label, size, fill=color, weight=weight))
    elif label is not None:
        lab = label if isinstance(label, (list, tuple)) else [label]
        sb = sub if isinstance(sub, (list, tuple)) else [sub]
        lab_h = size * 1.3 * len(lab)
        sub_h = sub_size * 1.35 * len(sb)
        top = y + h / 2 - (lab_h + sub_h + 4) / 2
        parts.append(ctext(x + w / 2, top + lab_h / 2, lab, size, fill=color, weight=weight, lh=1.3))
        parts.append(ctext(x + w / 2, top + lab_h + 4 + sub_h / 2, sb, sub_size, fill=sub_color, lh=1.35))
    return "".join(parts)


def line(x1, y1, x2, y2, stroke=INK, sw=1.5, dash=None, opacity=None):
    a = f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return f"<line {a}/>"


_MARKERS = {}


def _marker_id(color):
    mid = "ah_" + color.strip("#")
    _MARKERS[mid] = color
    return mid


def path(d, stroke=INK, sw=1.8, fill="none", arrow=False, dash=None, opacity=None, start_arrow=False):
    a = f'd="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'
    if arrow:
        a += f' marker-end="url(#{_marker_id(stroke)})"'
    if start_arrow:
        a += f' marker-start="url(#{_marker_id(stroke)}_s)"'
        _MARKERS[_marker_id(stroke) + "_s"] = stroke
    if dash:
        a += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return f"<path {a}/>"


def arrow(x1, y1, x2, y2, stroke=INK, sw=1.8, dash=None):
    return path(f"M{x1},{y1} L{x2},{y2}", stroke, sw, arrow=True, dash=dash)


def poly_arrow(points, stroke=INK, sw=1.8, dash=None, r=10):
    """Orthogonal polyline with rounded corners and an arrow head."""
    d = f"M{points[0][0]},{points[0][1]}"
    for i in range(1, len(points) - 1):
        (x0, y0), (x1, y1), (x2, y2) = points[i - 1], points[i], points[i + 1]
        l1 = math.hypot(x1 - x0, y1 - y0)
        l2 = math.hypot(x2 - x1, y2 - y1)
        rr = min(r, l1 / 2, l2 / 2)
        ax = x1 - (x1 - x0) / l1 * rr if l1 else x1
        ay = y1 - (y1 - y0) / l1 * rr if l1 else y1
        bx = x1 + (x2 - x1) / l2 * rr if l2 else x1
        by = y1 + (y2 - y1) / l2 * rr if l2 else y1
        d += f" L{ax:.1f},{ay:.1f} Q{x1},{y1} {bx:.1f},{by:.1f}"
    d += f" L{points[-1][0]},{points[-1][1]}"
    return path(d, stroke, sw, arrow=True, dash=dash)


def circle(cx, cy, r, fill="#FFFFFF", stroke=INK, sw=1.5):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def plus_node(cx, cy, r=13, stroke=INK, fill="#FFFFFF"):
    return (circle(cx, cy, r, fill, stroke, 1.6) + line(cx - r * 0.55, cy, cx + r * 0.55, cy, stroke, 1.8)
            + line(cx, cy - r * 0.55, cx, cy + r * 0.55, stroke, 1.8))


def mix(c1, c2, t):
    """Linear blend of two hex colors, t in [0,1]."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02X}" for i in range(3))


def grid(x, y, rows, cols, cell, fill_fn=None, stroke="#FFFFFF", sw=1.2, values=None,
         vsize=12, vcolor_fn=None, cell_h=None):
    ch = cell_h or cell
    parts = []
    for r in range(rows):
        for c in range(cols):
            f = fill_fn(r, c) if fill_fn else BLUE_L
            parts.append(f'<rect x="{x + c * cell}" y="{y + r * ch}" width="{cell}" height="{ch}" '
                         f'fill="{f}" stroke="{stroke}" stroke-width="{sw}"/>')
            if values is not None:
                v = values[r][c]
                col = vcolor_fn(r, c) if vcolor_fn else INK
                parts.append(ctext(x + c * cell + cell / 2, y + r * ch + ch / 2, v, vsize, fill=col, family=FONT))
    return "".join(parts)


def chip(x, y, w, h, s, fill=PANEL2, stroke=LINE, color=INK, size=16, weight=500, rx=6):
    return rect(x, y, w, h, fill, stroke, 1.2, rx) + ctext(x + w / 2, y + h / 2, s, size, fill=color, weight=weight)


def tag(x, y, s, fill=BLUE, color="#FFFFFF", size=14, pad=9, h=24):
    w = estimate_width(s, size) + pad * 2
    return rect(x, y, w, h, fill, fill, 1, h / 2) + ctext(x + w / 2, y + h / 2, s, size, fill=color, weight=600)


def estimate_width(s, size):
    s = re.sub(r"[_^]\{([^}]*)\}", r"\1", s).replace("**", "")
    w = 0
    for ch in s:
        w += size * (1.0 if ord(ch) > 0x2E80 else 0.58)
    return w


def step_badge(x, y, n, fill=INK, r=14):
    return circle(x, y, r, fill, fill, 1) + ctext(x, y, str(n), r * 1.1, fill="#FFFFFF", weight=700)


def header(title, subtitle, page, kicker):
    p = []
    p.append(text(60, 62, kicker, 17, fill=MUTED, weight=700))
    p.append(text(60, 102, title, 34, fill=INK, weight=700))
    p.append(text(60, 134, subtitle, 18, fill=MUTED))
    p.append(line(60, 152, W - 60, 152, LINE, 1.2))
    return "".join(p)


def footer(page):
    p = []
    p.append(text(60, H - 22, SOURCE + " · 数值取自原文", 13, fill=FAINT))
    p.append(text(W - 60, H - 22, f"{page:02d} / {TOTAL_PAGES:02d}", 15, fill=MUTED, weight=600, anchor="end"))
    return "".join(p)


def document(body):
    _markers = []
    for mid, color in _MARKERS.items():
        if mid.endswith("_s"):
            _markers.append(
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
        else:
            _markers.append(
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<defs>{"".join(_markers)}</defs>'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>{body}</svg>')


def render(name, body, scale=2):
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.dirname(here)
    svg_path = os.path.join(here, f"{name}.svg")
    png_path = os.path.join(outdir, f"{name}.png")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(document(body))
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        sys.exit("no chrome found")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
                    f"--force-device-scale-factor={scale}", f"--window-size={W},{H}",
                    "--default-background-color=FFFFFFFF",
                    f"--screenshot={png_path}", "file://" + svg_path],
                   check=True, capture_output=True)
    print("wrote", png_path)


def segs(x, y, parts, size=18, family=FONT, anchor="start"):
    """One text line made of differently styled segments: [(text, color, weight), ...]."""
    out = []
    for p in parts:
        s, col, wt = (p + (INK, 400))[:3] if isinstance(p, tuple) else (p, INK, 400)
        out.append(f'<tspan fill="{col}" font-weight="{wt}">{rich(s, size)}</tspan>')
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'text-anchor="{anchor}">{"".join(out)}</text>')


def legend(x, y, items=("lm", "prompt", "base"), size=13):
    """One-line colour legend; x is the right edge (anchor end)."""
    spec = {"lm": (LM, LM_L, "语言模型 GPT-2"), "prompt": (PROMPT, PROMPT_L, "任务提示"),
            "base": (BASE, BASE_L, "数据 / baseline")}
    parts, cx = [], x
    for key in reversed(items):
        col, fill, lab = spec[key]
        w = estimate_width(lab, size)
        cx -= w
        parts.append(text(cx, y + 5, lab, size, fill=col if key != "base" else MUTED, weight=600))
        cx -= 22
        parts.append(rect(cx, y - 7, 15, 15, fill, col, 1.4, 3))
        cx -= 22
    return "".join(parts)


def prompt_seq(x, y, parts, h=34, size=14, gap=6):
    """Token-sequence strip: parts = [(text, kind)]. Returns (svg, end_x).

    kind: "ctx" = input (grey), "prompt" = task prompt (orange),
          "out" = text GPT-2 generates (blue, dashed),
          "slot" = answer part of a data example, not generated by GPT-2 (grey, dashed).
    """
    out, cx = [], x
    for s, kind in parts:
        w = estimate_width(s, size) + 22
        if kind == "prompt":
            out.append(chip(cx, y, w, h, s, PROMPT_L, PROMPT, PROMPT, size, 700))
        elif kind == "out":
            out.append(chip(cx, y, w, h, s, LM_LL, LM, LM, size, 600, rx=6).replace(
                f'stroke="{LM}" stroke-width="1.2"', f'stroke="{LM}" stroke-width="1.2" stroke-dasharray="4 3"'))
        elif kind == "slot":
            out.append(rect(cx, y, w, h, "#FFFFFF", MUTED, 1.2, 6, dash="4 3")
                       + ctext(cx + w / 2, y + h / 2, s, size, fill=INK, weight=600))
        else:
            out.append(chip(cx, y, w, h, s, PANEL2, LINE, INK, size, 500))
        cx += w + gap
    return "".join(out), cx
