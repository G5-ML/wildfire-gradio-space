---
title: Wildfire Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
---

# 🔥 Wildfire Detection (EfficientNetV2B0 + Grad-CAM)

Upload a satellite/aerial image and get:
- **Wildfire / No-wildfire** verdict
- **Confidence score** (the model's own probability, not a tuned operating
  threshold — see note below)
- **Grad-CAM heatmap** showing which part of the image drove the decision

## 📁 Files

```
.
├── app.py              # Gradio UI + inference
├── gradcam.py           # Grad-CAM implementation (auto-detects the last
│                         conv layer, works for any backbone built the way
│                         model_utils.py builds it)
├── requirements.txt
├── README.md
└── model/
    ├── EfficientNetV2B0_wildfire.keras   # <-- YOU add this file
    └── threshold.json                     # <-- optional, see below
```

## ⚠️ Before deploying: add your trained model

This repo does **not** include a trained model — you need to drop your own
`.keras` file at:

```
model/EfficientNetV2B0_wildfire.keras
```

From the training project, that's the file saved during/after
`train_utils.train_two_stage` — e.g. copy in your best fine-tuned
checkpoint:

```
<ARTIFACT_DRIVE_DIR>/EfficientNetV2B0/checkpoints/stage2_finetune_best.keras
```

(rename it to `EfficientNetV2B0_wildfire.keras`, or set the `MODEL_PATH`
environment variable in your Space's settings to point at whatever filename
you use instead).

### Optional: `threshold.json`

If you also drop a `model/threshold.json` file in (from your F2-optimized
threshold search), the app will use it as the **decision boundary** — which
side of the probability line counts as "wildfire" — instead of the default
0.5 cutoff. It's read once at startup by `_load_decision_threshold()` in
`app.py`, which accepts any of these shapes:

```json
{"best_threshold": 0.37}
{"threshold": 0.37}
{"EfficientNetV2B0": {"best_threshold": 0.37}}
0.37
```

It checks `best_threshold`, `threshold`, `optimal_threshold`, `f2_threshold`,
and `value` as key names (optionally nested under the model name). If your
file uses a different shape, either rename the key or edit the `for key in
(...)` line in `_load_decision_threshold`. **Check your Space's logs after
first launch** — it prints exactly which threshold it loaded (or why it
fell back to 0.5), so a schema mismatch is never silent.

Note this only shifts *where the line is drawn* for the WILDFIRE /
NO WILDFIRE label. The confidence percentage shown to users is always the
model's raw probability for whichever label got picked — never the
threshold value itself.

**Model file is large (~25–30 MB)** — if you deploy via `git push` rather
than the Spaces web UI drag-and-drop, track it with Git LFS first:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
```

## 🚀 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   - **SDK**: Gradio
   - **Hardware**: CPU basic is enough for inference (no training happens here)
2. Upload all files in this folder (`app.py`, `gradcam.py`, `requirements.txt`,
   `README.md`) plus your `model/EfficientNetV2B0_wildfire.keras`, either by:
   - dragging them into the Space's "Files" tab in the browser, or
   - `git clone` your Space's repo, copy these files in, `git add . && git commit -m "deploy" && git push`
3. The Space will build automatically (installs `requirements.txt`, then runs
   `app.py`). First build can take a few minutes because of the TensorFlow
   install.

## 🖥️ Run locally first (recommended)

```bash
pip install -r requirements.txt
python app.py
```

Then open the printed local URL (usually `http://127.0.0.1:7860`).

## How it works

- **Preprocessing**: image resized to 224×224, RGB, raw `[0, 255]` pixel
  values. EfficientNetV2's rescaling/normalization is baked into the model
  itself as its first layers (`include_preprocessing=True`), so the app
  deliberately does **not** rescale to `[0, 1]` or apply ImageNet mean/std
  manually — doing so would double-preprocess and silently break predictions.
- **Grad-CAM**: `gradcam.py` auto-detects the last 4D (spatial) layer in the
  model — no hardcoded layer name — computes gradients of the sigmoid output
  w.r.t. that layer's activations, pools them into per-channel importance
  weights, and produces a heatmap. This works generically for EfficientNetV2,
  EfficientNet, ConvNeXt, or any backbone built the same way (see the
  training project's `model_utils.py` for why that's possible).
- **Confidence score**: the model outputs one sigmoid value, `P(wildfire)`.
  The label uses a standard 0.5 cutoff; the displayed confidence is that
  probability (or its complement) for whichever label was predicted — i.e.
  "how sure is the model," in plain terms a viewer can interpret at a glance.

## Limitations

This is a research/demo tool trained on one public satellite-imagery dataset
(Quebec, Canada wildfire prediction dataset). It has not been validated for
operational wildfire detection, and should not be used for real safety or
emergency-response decisions.
