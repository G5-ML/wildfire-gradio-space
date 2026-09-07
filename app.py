"""
Wildfire Detection - Gradio Space / Colab Deployment
=====================================================
Upload a satellite/aerial image -> EfficientNetV2B0 classifies it as
wildfire / no-wildfire, with a Grad-CAM heatmap showing which region of
the image the model actually looked at.
"""

import json
import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_spatial_layer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/EfficientNetV2B0_wildfire.keras")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "model/threshold.json")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["nowildfire", "wildfire"]
MODEL_NAME = "EfficientNetV2B0"


def _load_decision_threshold(default: float = 0.5) -> float:
    if not os.path.exists(THRESHOLD_PATH):
        print(f"[startup] No threshold file at '{THRESHOLD_PATH}' - using default {default}.")
        return default

    try:
        with open(THRESHOLD_PATH) as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[startup] Could not parse '{THRESHOLD_PATH}' ({e}) - using default {default}.")
        return default

    candidate = data
    if isinstance(data, dict) and MODEL_NAME in data:
        candidate = data[MODEL_NAME]

    for key in ("best_threshold", "threshold", "optimal_threshold", "f2_threshold", "value"):
        if isinstance(candidate, dict) and key in candidate:
            val = float(candidate[key])
            print(f"[startup] Using decision threshold {val} from '{THRESHOLD_PATH}' (key='{key}').")
            return val

    if isinstance(candidate, (int, float)):
        val = float(candidate)
        print(f"[startup] Using decision threshold {val} from '{THRESHOLD_PATH}' (bare number).")
        return val

    print(f"[startup] WARNING: '{THRESHOLD_PATH}' fallback to default {default}.")
    return default


DECISION_THRESHOLD = _load_decision_threshold(default=0.5)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
_model = None
_model_error = None
_target_layer = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model file found at '{MODEL_PATH}'. Place your trained "
            f".keras file there before running."
        )
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _target_layer = find_last_spatial_layer(_model)
    print(f"[startup] Loaded model from '{MODEL_PATH}'. "
          f"Grad-CAM target layer: '{_target_layer}'.")
except Exception as e:  # noqa: BLE001
    _model_error = str(e)
    print(f"[startup] ERROR loading model: {_model_error}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def _verdict_card(prob_wildfire: float) -> str:
    is_wildfire = prob_wildfire >= DECISION_THRESHOLD
    confidence = prob_wildfire if is_wildfire else (1 - prob_wildfire)
    pct = confidence * 100

    if is_wildfire:
        label, css_class, icon = "WILDFIRE DETECTED", "verdict-fire", "🔥"
    else:
        label, css_class, icon = "NO WILDFIRE DETECTED", "verdict-safe", "✅"

    note = ""
    if is_wildfire and prob_wildfire < 0.5:
        note = (
            '<div class="verdict-note">Flagged cautiously - this model is '
            "tuned to favor catching real fires over avoiding false "
            "alarms.</div>"
        )

    return f"""
    <div class="verdict-card {css_class}">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-label">{label}</div>
        <div class="verdict-confidence">{pct:.1f}% confidence</div>
        {note}
    </div>
    """


def _likelihood_bar(prob_wildfire: float) -> str:
    pct = prob_wildfire * 100
    threshold_pct = DECISION_THRESHOLD * 100
    return f"""
    <div class="likelihood-block">
        <div class="likelihood-row">
            <span>Wildfire likelihood</span>
            <span class="likelihood-pct">{pct:.1f}%</span>
        </div>
        <div class="likelihood-track">
            <div class="likelihood-fill" style="width:{pct:.2f}%;"></div>
            <div class="likelihood-marker" style="left:{threshold_pct:.2f}%;"
                 title="Alert line - the model flags anything past this point"></div>
        </div>
        <div class="likelihood-scale">
            <span>No wildfire</span>
            <span>Wildfire</span>
        </div>
    </div>
    """


def predict(image: Image.Image):
    if image is None:
        return (
            "<div class='hint-card'>Upload an image to get started.</div>",
            "",
            None,
        )

    if _model is None:
        return (
            f"<div class='error-card'>⚠️ Model not loaded: {_model_error}</div>",
            "",
            None,
        )

    img_batch = _preprocess(image)
    heatmap, prob_wildfire = make_gradcam_heatmap(img_batch, _model, _target_layer)
    overlay = overlay_heatmap(heatmap, image, alpha=0.45, colormap_name="inferno")

    return _verdict_card(prob_wildfire), _likelihood_bar(prob_wildfire), overlay


# ---------------------------------------------------------------------------
# UI - dark, wildfire-themed
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --ember-orange: #ff6a3d;
    --ember-gold: #ffb347;
    --smoke-dark: #14100e;
    --smoke-panel: #201a17;
    --smoke-panel-2: #241d19;
    --smoke-border: #3a2a22;
    --smoke-text: #f2e9e4;
    --smoke-muted: #b8a89f;
    --safe-green: #4ade80;
    --fire-red: #ff5252;
}

.gradio-container {
    background: radial-gradient(circle at 15% -10%, #2c1810 0%, #120d0b 55%) !important;
    color: var(--smoke-text) !important;
}

#app-title { text-align: center; padding: 8px 0 0 0; }
#app-title h1 {
    font-size: 2.3rem; font-weight: 800; margin-bottom: 2px;
    background: linear-gradient(90deg, var(--ember-gold), var(--ember-orange) 60%, #d7263d);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
#app-subtitle { text-align: center; color: var(--smoke-muted) !important; margin-top: -6px; }

.panel {
    background: var(--smoke-panel) !important;
    border: 1px solid var(--smoke-border) !important;
    border-radius: 18px !important; padding: 14px !important;
}

#analyze-btn {
    background: linear-gradient(90deg, var(--ember-orange), var(--ember-gold)) !important;
    border: none !important; color: #1a1210 !important;
    font-weight: 700 !important; font-size: 1.02rem !important;
    box-shadow: 0 6px 18px rgba(255, 106, 61, 0.25);
}

.verdict-card {
    text-align: center; border-radius: 18px; padding: 22px 16px;
    border: 1px solid var(--smoke-border); background: var(--smoke-panel-2);
}
.verdict-icon { font-size: 2.4rem; line-height: 1; margin-bottom: 4px; }
.verdict-label { font-size: 1.35rem; font-weight: 800; letter-spacing: 0.02em; }
.verdict-confidence { color: var(--smoke-muted); margin-top: 4px; font-size: 0.95rem; }
.verdict-card.verdict-fire .verdict-label { color: var(--fire-red); }
.verdict-card.verdict-safe .verdict-label { color: var(--safe-green); }
.verdict-note {
    margin-top: 10px; font-size: 0.8rem; color: var(--smoke-muted);
    border-top: 1px dashed var(--smoke-border); padding-top: 8px;
}

.likelihood-block { padding: 4px 6px; }
.likelihood-row {
    display: flex; justify-content: space-between;
    font-size: 0.9rem; color: var(--smoke-muted); margin-bottom: 6px;
}
.likelihood-pct { color: var(--smoke-text); font-weight: 700; }
.likelihood-track {
    position: relative; width: 100%; height: 14px; border-radius: 999px;
    background: #2b2320; overflow: visible; border: 1px solid var(--smoke-border);
}
.likelihood-fill {
    height: 100%; border-radius: 999px; overflow: hidden;
    background: linear-gradient(90deg, #4ade80 0%, #ffb347 55%, #ff5252 100%);
    transition: width 0.4s ease;
}
.likelihood-marker {
    position: absolute; top: -3px; bottom: -3px; width: 2px;
    background: var(--smoke-text); opacity: 0.85; transform: translateX(-1px);
}
.likelihood-marker::after {
    content: ""; position: absolute; left: 50%; top: -4px;
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--smoke-text); transform: translateX(-50%);
}
.likelihood-scale {
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: var(--smoke-muted); margin-top: 4px;
}

.hint-card, .error-card {
    text-align: center; padding: 24px; border-radius: 14px;
    border: 1px dashed var(--smoke-border); color: var(--smoke-muted);
}
.error-card { color: var(--fire-red); border-color: var(--fire-red); }

#footer-note {
    text-align: center; color: var(--smoke-muted) !important;
    font-size: 0.82rem; margin-top: 6px;
}
"""

with gr.Blocks(
    title="Wildfire Detection - EfficientNetV2B0",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(primary_hue="orange", neutral_hue="stone"),
) as demo:
    with gr.Column(elem_id="app-title"):
        gr.Markdown("# 🔥 Wildfire Detection")
        gr.Markdown(
            "EfficientNetV2B0 classifier with Grad-CAM explainability",
            elem_id="app-subtitle",
        )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel"):
                image_input = gr.Image(
                    type="pil",
                    label="Upload a satellite / aerial image",
                    height=340,
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Image", elem_id="analyze-btn", size="lg"
                )

        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel"):
                verdict_output = gr.HTML(
                    "<div class='hint-card'>Upload an image to get started.</div>"
                )
                likelihood_output = gr.HTML("")

    with gr.Group(elem_classes="panel"):
        gr.Markdown("### 🌡️ Grad-CAM Heatmap")
        gr.Markdown(
            "Warmer colors show the image regions that most influenced the "
            "model's decision.",
            elem_id="footer-note",
        )
        gradcam_output = gr.Image(label="Grad-CAM overlay", height=420)

    with gr.Accordion("ℹ️ How this works / limitations", open=False):
        gr.Markdown(
            """
- **Model**: EfficientNetV2B0, fine-tuned on the Wildfire Prediction Dataset.
- **Grad-CAM**: Highlights pixels the last conv layer relied on most.
- **Confidence score**: Raw probability P(wildfire).
- **Research/demo tool only**: Not validated for operational emergency response.
            """
        )

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[verdict_output, likelihood_output, gradcam_output],
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[verdict_output, likelihood_output, gradcam_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
