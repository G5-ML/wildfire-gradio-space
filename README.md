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
