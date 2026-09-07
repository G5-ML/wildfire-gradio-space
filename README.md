---

## title: Wildfire Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit

# 🔥 Wildfire Detection (EfficientNetV2B0 + Grad-CAM)

Upload a satellite/aerial image and get:

* **Wildfire / No-wildfire** verdict
* **Confidence score** (the model's own probability, aligned with your tuned threshold)
* **Grad-CAM heatmap** showing which part of the image drove the decision

## 📁 Files

```
.
├── app.py              # Gradio UI + inference
├── gradcam.py           # Grad-CAM implementation (auto-detects the last conv layer)
├── requirements.txt
├── README.md
└── model/
    ├── stage2_finetune_best.keras   # Trained model weights
    └── threshold.json               # F2-optimized operating threshold

```

## 🚀 Run via Google Colab & Gradio Share

Because this demo runs on a free Google Colab notebook with a T4 GPU using a public shareable URL, you don't need manual Hugging Face cloud deployments or paid hardware tiers.

1. Open a new notebook in **[Google Colab](https://colab.research.google.com/)** and set **Runtime -> Change runtime type** to **T4 GPU**.
2. Run the deployment block in your notebook:

```bash
# Clone repository
!git clone https://github.com/G5-ML/wildfire-gradio-space.git wildfire_app
%cd wildfire_app

# Install required dependencies
!pip install -q gradio tensorflow opencv-python matplotlib

# Set model path and launch app with a public shareable URL
%env MODEL_PATH=model/stage2_finetune_best.keras
!python app.py

```

3. Colab will output a public URL (`[https://xxxx.gradio.live](https://xxxx.gradio.live)`) that stays active for up to 72 hours.

## How it works

* **Preprocessing**: Image resized to 224×224, RGB, raw `[0, 255]` pixel values. EfficientNetV2's normalization is baked into the model architecture itself (`include_preprocessing=True`).
* **Grad-CAM**: `gradcam.py` auto-detects the last spatial layer, computes gradients of the output with respect to feature maps, and creates an interpretable heatmap overlay.
* **Decision Threshold**: Automatically loads F2-optimized thresholds from `threshold.json` to balance recall and precision for wildfire detection.

## Limitations

This is a research/demo tool trained on a public satellite-imagery dataset (Quebec, Canada wildfire prediction dataset). It has not been validated for operational emergency response and must not be used for real-world safety decisions.
