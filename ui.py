"""
PyroVision - presentation layer.

Everything visual lives here so `app.py` stays about the model: the Gradio
theme, the webfont <head>, the animated ambient background, and the small
HTML fragments the app renders into `gr.HTML` slots.

Only CSS/SVG animation is used - Gradio renders component HTML with Svelte's
`{@html ...}`, which applies <style>/<svg> but never executes <script>, so
anything that moves has to move without JS.
"""

from __future__ import annotations

import base64
import inspect
import io
import math
import random
from html import escape
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "wildfire-sentinel-logo.png"
CSS_PATH = ROOT / "style.css"

BRAND = "PyroVision"
TAGLINE = "Orbital wildfire intelligence"


# ---------------------------------------------------------------------------
# Brand mark
# ---------------------------------------------------------------------------
def _logo_data_uri(size: int = 160) -> str:
    """Inline the logo as a data URI.

    The source PNG is ~1 MB, which is far too heavy to ship on every page
    load for a 46px mark - downscale it once at import time instead. Falls
    back to an empty string so a missing asset degrades to the flame glyph
    rather than a broken image.
    """
    if not LOGO_PATH.exists():
        print(f"[ui] Logo not found at '{LOGO_PATH}' - using the text mark only.")
        return ""

    with Image.open(LOGO_PATH) as img:
        img = img.convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)

    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


LOGO_URI = _logo_data_uri()


def _mark(alt: str = "") -> str:
    if not LOGO_URI:
        return '<span aria-hidden="true">&#128293;</span>'
    return f'<img src="{LOGO_URI}" alt="{escape(alt)}" />'


# ---------------------------------------------------------------------------
# Icons - thin-stroke line set, one visual language across the whole UI
# ---------------------------------------------------------------------------
_PATHS = {
    "flame": (
        '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 '
        '2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 '
        '1-3a2.5 2.5 0 0 0 2.5 2.5z"/>'
    ),
    "shield": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 '
        '13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 '
        '3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>'
    ),
    "radar": (
        '<path d="M19.07 4.93A10 10 0 0 0 6.99 3.34"/><path d="M4 6h.01"/>'
        '<path d="M2.29 9.62A10 10 0 1 0 21.31 8.35"/>'
        '<path d="M16.24 7.76A6 6 0 1 0 8.23 16.67"/><path d="M12 18h.01"/>'
        '<path d="M17.99 11.66A6 6 0 0 1 15.77 16.67"/>'
        '<path d="m13.41 10.59 5.66-5.66"/><circle cx="12" cy="12" r="2"/>'
    ),
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<path d="M7 12h10"/>'
    ),
    "layers": (
        '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 '
        '3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>'
        '<path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>'
        '<path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "image": (
        '<rect width="18" height="18" x="3" y="3" rx="3"/>'
        '<circle cx="9" cy="9" r="1.6"/>'
        '<path d="m21 15-3.09-3.09a2 2 0 0 0-2.83 0L6 21"/>'
    ),
    "gauge": (
        '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
    ),
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 '
        '0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "slide": (
        '<path d="m18 8 4 4-4 4"/><path d="M2 12h20"/><path d="m6 8-4 4 4 4"/>'
    ),
    "bolt": (
        '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 '
        '6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 '
        '1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>'
    ),
}


def icon(name: str, size: int = 24) -> str:
    """Inline SVG icon. Colour is inherited via `currentColor`."""
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{_PATHS[name]}</svg>'
    )


# ---------------------------------------------------------------------------
# Theme + <head>
# ---------------------------------------------------------------------------
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2\
?family=Sora:wght@400;600;700;800\
&family=Inter:wght@400;500;600;700\
&family=JetBrains+Mono:wght@500;700\
&display=swap">
<meta name="theme-color" content="#2C0D07">
<meta name="description" content="PyroVision - wildfire detection and Grad-CAM \
explainability for satellite and aerial imagery.">
"""

_DISPLAY = '"Sora", "Segoe UI", system-ui, -apple-system, sans-serif'
_BODY = '"Inter", "Segoe UI", system-ui, -apple-system, sans-serif'
_MONO = '"JetBrains Mono", ui-monospace, "SFMono-Regular", Consolas, monospace'

# Brand gradient, repeated here because theme tokens can't read CSS vars that
# are declared in the stylesheet Gradio injects afterwards.
_BRAND_GRADIENT = (
    "linear-gradient(100deg, #FFDF8C 0%, #FFC24A 20%, #FF8B3D 50%, "
    "#FF5A2B 75%, #E8402A 100%)"
)


def pyrovision_theme() -> gr.themes.Base:
    """Gradio theme tokens for the Ember Watch palette.

    Light and dark variants are set to the same values on purpose: the app is
    a single committed dark design, so it must not shift with the visitor's
    OS colour-scheme preference.
    """
    # Not every token has a `_dark` twin, and `set()` rejects unknown keyword
    # arguments outright - so ask it what it accepts rather than guessing.
    accepted = set(inspect.signature(gr.themes.Base.set).parameters)
    both: dict[str, str] = {}

    def pair(**kwargs):
        """Set a token and, where one exists, its `_dark` twin to the same value."""
        for key, value in kwargs.items():
            for name in (key, f"{key}_dark"):
                if name in accepted:
                    both[name] = value

    pair(
        body_background_fill="#0A0503",
        background_fill_primary="rgba(32, 11, 8, 0.55)",
        background_fill_secondary="rgba(20, 8, 6, 0.42)",
        block_background_fill="rgba(32, 11, 8, 0.50)",
        block_border_color="rgba(255, 196, 150, 0.15)",
        block_label_background_fill="rgba(20, 8, 6, 0.80)",
        block_label_border_color="rgba(255, 196, 150, 0.15)",
        block_label_text_color="rgba(255, 243, 234, 0.54)",
        block_title_text_color="rgba(255, 243, 234, 0.74)",
        block_info_text_color="rgba(255, 243, 234, 0.38)",
        body_text_color="#FFF3EA",
        body_text_color_subdued="rgba(255, 243, 234, 0.54)",
        border_color_primary="rgba(255, 196, 150, 0.15)",
        border_color_accent="rgba(255, 194, 74, 0.45)",
        border_color_accent_subdued="rgba(255, 194, 74, 0.25)",
        color_accent="#FF6B2C",
        color_accent_soft="rgba(255, 107, 44, 0.14)",
        panel_background_fill="rgba(20, 8, 6, 0.45)",
        panel_border_color="rgba(255, 196, 150, 0.15)",
        input_background_fill="rgba(16, 7, 5, 0.55)",
        input_background_fill_focus="rgba(24, 10, 7, 0.70)",
        input_background_fill_hover="rgba(22, 9, 7, 0.65)",
        input_border_color="rgba(255, 196, 150, 0.16)",
        input_border_color_focus="rgba(255, 194, 74, 0.55)",
        input_border_color_hover="rgba(255, 196, 150, 0.30)",
        input_placeholder_color="rgba(255, 243, 234, 0.38)",
        button_primary_background_fill=_BRAND_GRADIENT,
        button_primary_background_fill_hover=_BRAND_GRADIENT,
        button_primary_border_color="transparent",
        button_primary_border_color_hover="transparent",
        button_primary_text_color="#230903",
        button_primary_text_color_hover="#230903",
        button_secondary_background_fill="rgba(255, 170, 90, 0.06)",
        button_secondary_background_fill_hover="rgba(255, 170, 90, 0.12)",
        button_secondary_border_color="rgba(255, 196, 150, 0.28)",
        button_secondary_border_color_hover="rgba(255, 196, 150, 0.45)",
        button_secondary_text_color="rgba(255, 243, 234, 0.82)",
        button_secondary_text_color_hover="#FFF3EA",
        button_cancel_background_fill="rgba(232, 64, 42, 0.14)",
        button_cancel_background_fill_hover="rgba(232, 64, 42, 0.24)",
        button_cancel_border_color="rgba(232, 64, 42, 0.45)",
        button_cancel_text_color="#FFC9BE",
        checkbox_background_color="rgba(16, 7, 5, 0.6)",
        checkbox_background_color_selected="#FF6B2C",
        checkbox_border_color="rgba(255, 196, 150, 0.30)",
        checkbox_border_color_focus="#FFC24A",
        checkbox_border_color_selected="#FF6B2C",
        checkbox_label_background_fill="rgba(255, 170, 90, 0.06)",
        checkbox_label_background_fill_selected="rgba(255, 107, 44, 0.16)",
        checkbox_label_border_color="rgba(255, 196, 150, 0.18)",
        checkbox_label_text_color="rgba(255, 243, 234, 0.82)",
        slider_color="#FF6B2C",
        loader_color="#FFC24A",
        link_text_color="#FFC24A",
        link_text_color_hover="#FFDB8F",
        link_text_color_active="#FFDB8F",
        link_text_color_visited="#FFC24A",
        accordion_text_color="#FFF3EA",
        stat_background_fill=_BRAND_GRADIENT,
        code_background_fill="rgba(16, 7, 5, 0.75)",
        error_background_fill="rgba(70, 14, 9, 0.55)",
        error_border_color="rgba(232, 64, 42, 0.45)",
        error_text_color="#FFC9BE",
        table_border_color="rgba(255, 196, 150, 0.15)",
        table_even_background_fill="rgba(20, 8, 6, 0.55)",
        table_odd_background_fill="rgba(28, 11, 8, 0.55)",
        table_text_color="#FFF3EA",
        shadow_drop="0 2px 10px -6px rgba(0, 0, 0, 0.9)",
        shadow_drop_lg="0 24px 60px -40px rgba(0, 0, 0, 0.95)",
    )

    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_lg,
        text_size=gr.themes.sizes.text_md,
        font=[_DISPLAY],
        font_mono=[_MONO],
    ).set(
        block_radius="22px",
        block_border_width="1px",
        block_label_radius="10px",
        input_radius="16px",
        button_large_radius="999px",
        button_medium_radius="999px",
        button_small_radius="999px",
        button_large_text_weight="600",
        button_medium_text_weight="600",
        button_transition="all 0.25s cubic-bezier(.22, 1, .36, 1)",
        button_transform_hover="translateY(-1px)",
        prose_text_weight="400",
        prose_header_text_weight="600",
        section_header_text_weight="600",
        layout_gap="18px",
        **both,
    )


# ---------------------------------------------------------------------------
# Ambient background
# ---------------------------------------------------------------------------
_VB_W, _VB_H = 2880, 900      # path space; drawn at 2x the tile width
_TILE = 1440                  # every wave period divides this, so it loops


def _ridge(y0: float, harmonics, step: int = 24) -> str:
    """One topographic-looking ridge line as an SVG path.

    `harmonics` are (cycles_per_tile, amplitude, phase) triples. Because every
    `cycles_per_tile` is an integer, f(x + _TILE) == f(x) - so translating the
    layer by exactly one tile loops seamlessly with no visible seam.
    """
    points = []
    for x in range(0, _VB_W + step, step):
        t = 2 * math.pi * x / _TILE
        y = y0 + sum(a * math.sin(k * t + p) for k, a, p in harmonics)
        points.append(f"{x} {y:.1f}")
    return "M" + "L".join(points)


def _ridge_layer(rng: random.Random, count: int, amp: float) -> str:
    paths = []
    for i in range(count):
        y0 = _VB_H * (i + 0.5) / count + rng.uniform(-24, 24)
        harmonics = [
            (1, amp * rng.uniform(0.7, 1.3), rng.uniform(0, math.tau)),
            (2, amp * rng.uniform(0.3, 0.6), rng.uniform(0, math.tau)),
            (3, amp * rng.uniform(0.2, 0.4), rng.uniform(0, math.tau)),
            (5, amp * rng.uniform(0.1, 0.2), rng.uniform(0, math.tau)),
        ]
        paths.append(f'<path d="{_ridge(y0, harmonics)}"/>')
    return "".join(paths)


def ambient_layer() -> str:
    """The fixed, non-interactive backdrop: drifting terrain contours, a slow
    satellite scan sweep, rising embers and a faint survey grid."""
    rng = random.Random(20250915)  # fixed seed -> identical markup every boot

    far = _ridge_layer(rng, 7, 46)
    near = _ridge_layer(rng, 5, 68)
    trace = _ridge_layer(rng, 2, 58)

    embers = "".join(
        '<i style="--x:{x:.1f}%;--d:{d:.0f}s;--delay:-{delay:.0f}s;'
        '--s:{s}px;--o:{o:.2f}"></i>'.format(
            x=rng.uniform(2, 98),
            d=rng.uniform(20, 38),
            delay=rng.uniform(0, 38),
            s=rng.choice([1, 2, 2, 3]),
            o=rng.uniform(0.14, 0.34),
        )
        for _ in range(18)
    )

    return f"""
<div class="pv-ambient" aria-hidden="true">
  <div class="pv-ambient__grid"></div>
  <svg class="pv-ambient__contours" viewBox="0 0 {_VB_W} {_VB_H}"
       preserveAspectRatio="none">
    <g class="pv-contours pv-contours--far">{far}</g>
    <g class="pv-contours pv-contours--near">{near}</g>
    <g class="pv-contours pv-contours--trace">{trace}</g>
  </svg>
  <div class="pv-ambient__scan"></div>
  <div class="pv-ambient__embers">{embers}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------
def topbar(*, model_ok: bool, model_name: str, threshold: float) -> str:
    if model_ok:
        status = (
            '<div class="pv-status">'
            '<span class="pv-status__dot"></span>'
            "<span>Model online</span>"
            f'<span class="pv-status__meta">{escape(model_name)} '
            f"&middot; t={threshold:.2f}</span>"
            "</div>"
        )
    else:
        status = (
            '<div class="pv-status pv-status--down">'
            '<span class="pv-status__dot"></span>'
            "<span>Model unavailable</span>"
            "</div>"
        )

    return f"""
<header class="pv-topbar">
  <div class="pv-brand">
    <span class="pv-brand__mark">{_mark(f"{BRAND} logo")}</span>
    <span class="pv-brand__text">
      <span class="pv-wordmark">{BRAND}</span>
      <span class="pv-brand__tag">{escape(TAGLINE)}</span>
    </span>
  </div>
  {status}
</header>
"""


def hero() -> str:
    pillars = [
        ("radar", "Detect", "EfficientNetV2B0, fine-tuned on Quebec wildfire "
                            "imagery, reads a frame in well under a second."),
        ("layers", "Explain", "Grad-CAM paints the exact pixels behind every "
                              "call, so you can check the model's reasoning."),
        ("gauge", "Calibrated", "The alert line is tuned for recall - "
                                "PyroVision would rather double-check than "
                                "miss a real fire."),
    ]
    cards = "".join(
        f"""
      <article class="pv-pillar">
        <span class="pv-pillar__icon">{icon(name, 18)}</span>
        <h3>{title}</h3>
        <p>{body}</p>
      </article>"""
        for name, title, body in pillars
    )

    return f"""
<section class="pv-hero">
  <span class="pv-eyebrow">{icon("bolt", 14)} Wildfire detection &amp; explainability</span>
  <h1>See the fire while it is still <em>a spark</em>.</h1>
  <p class="pv-hero__sub">
    Drop in a satellite or aerial frame. {BRAND} returns a wildfire verdict,
    a calibrated confidence score, and a heatmap of the evidence that drove
    the decision - no black box, no guesswork.
  </p>
  <div class="pv-pillars">{cards}</div>
</section>
"""


def rail() -> str:
    steps = [
        ("01", "Drop a frame", "JPG, PNG or WebP"),
        ("02", "Run the analysis", "One click, sub-second"),
        ("03", "Read the evidence", "Verdict plus Grad-CAM"),
    ]
    items = "".join(
        f"""
    <div class="pv-rail__step">
      <span class="pv-rail__num">{num}</span>
      <span class="pv-rail__body"><strong>{title}</strong><span>{sub}</span></span>
    </div>"""
        for num, title, sub in steps
    )
    return f'<div class="pv-rail">{items}</div>'


def card_head(icon_name: str, title: str, hint: str = "", chip: str = "") -> str:
    right = ""
    if chip:
        right = f'<span class="pv-chip">{escape(chip)}</span>'
    elif hint:
        right = f'<span class="pv-cardhead__hint">{escape(hint)}</span>'
    return f"""
<div class="pv-cardhead">
  <span class="pv-cardhead__title">{icon(icon_name, 17)}{escape(title)}</span>
  {right}
</div>
"""


def footer() -> str:
    return f"""
<footer class="pv-footer">
  <span class="pv-footer__brand">{_mark("")} {BRAND}</span>
  <span class="pv-footer__warn">
    {icon("alert", 14)} Research demo - not validated for emergency response
  </span>
</footer>
"""


# ---------------------------------------------------------------------------
# Result slots
# ---------------------------------------------------------------------------
def empty_state() -> str:
    return f"""
<div class="pv-empty">
  <span class="pv-empty__icon">{icon("target", 52)}</span>
  <h3>Awaiting a frame</h3>
  <p>
    Drop a satellite or aerial image into the panel on the left and
    {BRAND} will report back here.
  </p>
  <ul class="pv-empty__tips">
    <li>nadir imagery</li>
    <li>224px or larger</li>
    <li>jpg / png / webp</li>
  </ul>
</div>
"""


def scanning_state() -> str:
    """Shown the instant a frame arrives, before inference returns."""
    return """
<div class="pv-scan" role="status" aria-live="polite">
  <div class="pv-scan__reticle">
    <svg viewBox="0 0 120 120" aria-hidden="true">
      <circle class="pv-ret-ring" cx="60" cy="60" r="52"/>
      <circle class="pv-ret-ring" cx="60" cy="60" r="34"/>
      <circle class="pv-ret-ring pv-ret-ring--pulse" cx="60" cy="60" r="52"/>
      <path class="pv-ret-cross" d="M60 2v18M60 100v18M2 60h18M100 60h18"/>
      <g class="pv-ret-sweep">
        <path d="M60 60 L60 8 A52 52 0 0 1 105 34 Z"
              fill="url(#pvSweep)" opacity="0.55"/>
      </g>
      <circle class="pv-ret-core" cx="60" cy="60" r="4"/>
      <defs>
        <linearGradient id="pvSweep" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#FFC24A" stop-opacity="0.85"/>
          <stop offset="100%" stop-color="#FF6B2C" stop-opacity="0"/>
        </linearGradient>
      </defs>
    </svg>
  </div>
  <p class="pv-scan__title">
    Scanning frame<span class="pv-dots"><i></i><i></i><i></i></span>
  </p>
  <p class="pv-scan__steps">normalise &middot; classify &middot; grad-cam</p>
  <div class="pv-scan__bar"><i></i></div>
</div>
"""


def error_card(message: str) -> str:
    return f"""
<div class="pv-error" role="alert">
  {icon("alert", 20)}
  <div>
    <strong>The model could not be loaded</strong>
    Inference is unavailable until this is resolved - the rest of the
    interface still works.
    <code>{escape(message)}</code>
  </div>
</div>
"""


def verdict_card(prob: float, threshold: float) -> str:
    is_fire = prob >= threshold
    confidence = prob if is_fire else 1 - prob

    if is_fire:
        modifier, badge, label = "fire", "flame", "Wildfire detected"
    else:
        modifier, badge, label = "clear", "shield", "No wildfire detected"

    note = ""
    # A recall-tuned threshold can sit below 0.5, so the model can raise a
    # flag while still calling the frame "more likely clear". Say so plainly
    # rather than letting the number contradict the headline.
    if is_fire and prob < 0.5:
        note = f"""
  <p class="pv-verdict__note">
    {icon("info", 15)}
    <span>Flagged cautiously. This frame sits below even odds, but above the
    alert line - the threshold is tuned to favour catching real fires over
    avoiding false alarms.</span>
  </p>"""

    return f"""
<div class="pv-verdict pv-verdict--{modifier}" role="status" aria-live="polite">
  <div class="pv-verdict__aura" aria-hidden="true"></div>
  <div class="pv-verdict__row">
    <span class="pv-verdict__badge">{icon(badge, 26)}</span>
    <div>
      <p class="pv-verdict__eyebrow">Verdict</p>
      <h3 class="pv-verdict__label">{label}</h3>
    </div>
    <div class="pv-verdict__score">
      <span class="pv-verdict__num">{confidence * 100:.1f}<small>%</small></span>
      <span class="pv-verdict__num-label">confidence</span>
    </div>
  </div>{note}
</div>
"""


def meter(prob: float, threshold: float) -> str:
    pct = prob * 100
    thresh_pct = threshold * 100
    return f"""
<div class="pv-meter">
  <div class="pv-meter__head">
    <span class="pv-meter__title">Wildfire likelihood</span>
    <span class="pv-meter__value">{pct:.1f}%</span>
  </div>
  <div class="pv-meter__track" role="img"
       aria-label="Wildfire likelihood {pct:.1f} percent,
                   alert threshold {thresh_pct:.1f} percent">
    <div class="pv-meter__rest" style="--w:{pct:.2f}%"></div>
    <div class="pv-meter__thresh" style="--t:{thresh_pct:.2f}%"
         title="Alert line - PyroVision flags anything past this point"></div>
  </div>
  <div class="pv-meter__scale">
    <span>Clear</span>
    <b>alert line {thresh_pct:.1f}%</b>
    <span>Wildfire</span>
  </div>
</div>
"""


def metrics(prob: float, threshold: float, elapsed_s: float, model_name: str) -> str:
    margin = prob - threshold
    rows = [
        ("P(wildfire)", f"{prob:.3f}"),
        ("Alert line", f"{threshold:.3f}"),
        ("Margin", f"{margin:+.3f}"),
        ("Latency", f"{elapsed_s * 1000:.0f} ms"),
    ]
    cells = "".join(
        f'<div class="pv-metric"><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>'
        for k, v in rows
    )
    return (
        f'<dl class="pv-metrics">{cells}</dl>'
        f'<p class="pv-sr">Scored by {escape(model_name)}.</p>'
    )


def gradcam_placeholder() -> str:
    """Stands in for the comparison slider until there is something to show,
    so the evidence step is always visible but never an empty grey box."""
    return f"""
<div class="pv-empty pv-empty--wide">
  <span class="pv-empty__icon">{icon("layers", 46)}</span>
  <h3>No overlay yet</h3>
  <p>
    Once a frame has been analysed, the original and the Grad-CAM heatmap
    land here side by side, under a slider you can drag.
  </p>
</div>
"""


def gradcam_legend() -> str:
    return f"""
<div class="pv-legend">
  <span class="pv-legend__end">low attention</span>
  <span class="pv-legend__bar" role="img"
        aria-label="Grad-CAM colour scale, dark for low attention through to
                    bright yellow for high attention"></span>
  <span class="pv-legend__end">high attention</span>
</div>
<p class="pv-slider-hint">{icon("slide", 15)} Drag the handle to compare the
original frame with the heatmap</p>
"""
