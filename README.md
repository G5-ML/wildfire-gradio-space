---
title: PyroVision - Wildfire Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
---

# 🔥 PyroVision

**Wildfire detection and Grad-CAM explainability for satellite and aerial imagery.**

Drop in a frame and PyroVision returns three things:

* a **wildfire / no-wildfire verdict**
* a **calibrated confidence score**, aligned with the model's tuned operating threshold
* a **Grad-CAM heatmap** showing which pixels drove the decision

Under the hood it is an EfficientNetV2B0 fine-tuned on the Quebec wildfire
prediction dataset, served through Gradio.

## 📁 Files

```
.
├── app.py               # model loading, inference, Gradio layout
├── ui.py                # theme, webfonts, animated backdrop, HTML fragments
├── style.css            # the "Ember Watch" design system
├── gradcam.py           # Grad-CAM (auto-detects the last conv layer)
├── assets/
│   └── wildfire-sentinel-logo.png
├── model/
│   ├── stage2_finetune_best.keras   # trained weights
│   └── threshold.json               # F2-optimised operating threshold
├── requirements.txt
└── README.md
```

## 🚀 Run via Google Colab & Gradio Share

The demo runs on a free Colab T4 with a public share URL, so no Hugging Face
paid hardware tier is needed.

1. Open a notebook in **[Google Colab](https://colab.research.google.com/)** and
   set **Runtime → Change runtime type → T4 GPU**.
2. Run the deployment block:

```bash
# Clone repository
!git clone https://github.com/G5-ML/wildfire-gradio-space.git wildfire_app
%cd wildfire_app

# Install required dependencies
!pip install -q gradio tensorflow opencv-python matplotlib

# Set model path and launch with a public shareable URL
%env MODEL_PATH=model/stage2_finetune_best.keras
!python app.py
```

3. Colab prints a public URL (`https://xxxx.gradio.live`) that stays live for
   up to 72 hours.

## 🎨 The interface

The UI is a single committed dark design — it does not follow the visitor's
OS colour scheme.

**Palette — "Ember Watch."** Pulled from the brand mark: flame gold `#FFC24A`
→ orange `#FF6B2C` → crimson `#E8402A` over a deep ember ground, with the
logo's cyan orbit ring (`#3DDCE8`, and `#2FE0AE` for "clear") kept as the only
cool accent. Warm therefore always means danger and cool always means safe, so
the verdict never depends on hue alone.

**Type.** Sora for display and headings, Inter for UI text, JetBrains Mono for
numbers and telemetry. All three load from Google Fonts via `ui.HEAD`, with
system fallbacks.

**Motion.** The backdrop is drifting topographic contour lines, a slow
satellite scan sweep, rising embers and a faint survey grid — all under 12%
opacity. Uploads get a rotating ember rim; running an analysis shows a radar
reticle while inference streams. Every decorative animation stops under
`prefers-reduced-motion: reduce`.

### Working on the UI

Two things about Gradio are worth knowing before editing `style.css`:

* **Gradio re-serialises custom CSS through the CSSOM.** A `background:`
  shorthand whose value is a `var()` gets expanded into longhands, and the
  pending-substitution longhands come back *empty* — the gradient silently
  disappears. Use `background-image: var(--x)` instead. The same applies to
  prefixed `mask` / `mask-composite` pairs.
* **`gr.Group` draws its own border and background.** Panels use
  `gr.Column(elem_classes="pv-card")` so `.pv-card` is the only chrome on
  screen.

`gr.HTML` renders through Svelte's `{@html ...}`, which applies `<style>` and
`<svg>` but never executes `<script>` — so everything that moves does so with
CSS and SVG only.

## How it works

* **Preprocessing.** Frames are resized to 224×224 RGB and passed as raw
  `[0, 255]` pixels. EfficientNetV2's normalisation is baked into the
  architecture (`include_preprocessing=True`).
* **Grad-CAM.** `gradcam.py` auto-detects the last spatial layer, takes
  gradients of the wildfire score with respect to its feature maps, and blends
  the result over the frame at its native resolution.
* **Decision threshold.** Loaded from `threshold.json` and optimised for
  F-beta with beta=2. That favours recall, so a frame can be flagged while
  `P(wildfire)` is still below 0.5 — the UI says so explicitly when it happens.

## Limitations

This is a research and demonstration tool trained on one public
satellite-imagery dataset (Quebec, Canada). It has **not** been validated for
operational emergency response and must not be used for real-world safety
decisions. Smoke, cloud, sunset light and old burn scars are all known
confusers.
