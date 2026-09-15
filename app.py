"""
PyroVision - wildfire detection from satellite & aerial imagery.
================================================================
Upload a frame -> EfficientNetV2B0 classifies it as wildfire / no-wildfire,
with a Grad-CAM heatmap showing which region of the image the model actually
looked at.

Presentation (theme, CSS, HTML fragments) lives in `ui.py` / `style.css`.
"""

import json
import os
import time

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

import ui
from gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_spatial_layer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/stage2_finetune_best.keras")
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


def _no_overlay():
    """Evidence panel in its pre-result shape: slider away, placeholder up."""
    return gr.update(value=None, visible=False), gr.update(visible=True)


def predict(image: Image.Image):
    """Stream three UI states: idle -> scanning -> result.

    This is a generator so the scanning animation reaches the browser the
    moment the frame lands, instead of only after inference finishes.
    """
    if image is None:
        yield ui.empty_state(), "", "", *_no_overlay()
        return

    if _model is None:
        yield ui.error_card(_model_error or "unknown error"), "", "", *_no_overlay()
        return

    yield ui.scanning_state(), "", "", *_no_overlay()

    started = time.perf_counter()
    img_batch = _preprocess(image)
    heatmap, prob_wildfire = make_gradcam_heatmap(img_batch, _model, _target_layer)
    overlay = overlay_heatmap(heatmap, image, alpha=0.45, colormap_name="inferno")
    elapsed = time.perf_counter() - started

    yield (
        ui.verdict_card(prob_wildfire, DECISION_THRESHOLD),
        ui.meter(prob_wildfire, DECISION_THRESHOLD),
        ui.metrics(prob_wildfire, DECISION_THRESHOLD, elapsed, MODEL_NAME),
        gr.update(value=(image.convert("RGB"), overlay), visible=True),
        gr.update(visible=False),
    )


def reset():
    """Clear the frame and put every output slot back to its idle state."""
    return None, ui.empty_state(), "", "", *_no_overlay()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title=f"{ui.BRAND} - Wildfire Detection") as demo:
    # Fixed, non-interactive backdrop. Rendered first so it paints behind
    # everything else; #pv-app is lifted above it in style.css.
    gr.HTML(ui.ambient_layer(), elem_id="pv-ambient-host",
            container=False, padding=False)

    with gr.Column(elem_id="pv-app"):
        gr.HTML(
            ui.topbar(
                model_ok=_model is not None,
                model_name=MODEL_NAME,
                threshold=DECISION_THRESHOLD,
            ),
            container=False,
            padding=False,
        )
        gr.HTML(ui.hero(), container=False, padding=False)
        gr.HTML(ui.rail(), container=False, padding=False)

        with gr.Row(equal_height=False):
            # ---------------------------------------------------- input ---
            with gr.Column(scale=5):
                with gr.Column(elem_classes="pv-card"):
                    gr.HTML(
                        ui.card_head("image", "Source frame", chip="step 01"),
                        container=False,
                        padding=False,
                    )
                    with gr.Column(elem_classes="pv-drop", elem_id="pv-drop"):
                        image_input = gr.Image(
                            type="pil",
                            label="Satellite or aerial image",
                            show_label=False,
                            height=330,
                            sources=["upload", "clipboard"],
                            buttons=["fullscreen"],
                            container=False,
                            # "#" makes the first line a heading; Gradio
                            # renders this in place of its default copy.
                            placeholder=(
                                "# Drop a frame here\n"
                                "or click to browse your files"
                            ),
                        )
                    analyze_btn = gr.Button(
                        "Run wildfire analysis",
                        elem_id="pv-analyze",
                        size="lg",
                        variant="primary",
                    )
                    reset_btn = gr.Button(
                        "Clear frame", elem_id="pv-reset", size="lg"
                    )

            # --------------------------------------------------- output ---
            with gr.Column(scale=6):
                with gr.Column(elem_classes="pv-card"):
                    gr.HTML(
                        ui.card_head("gauge", "Assessment", chip="step 02"),
                        container=False,
                        padding=False,
                    )
                    verdict_output = gr.HTML(
                        ui.empty_state(), container=False, padding=False
                    )
                    meter_output = gr.HTML("", container=False, padding=False)
                    metrics_output = gr.HTML("", container=False, padding=False)

        # ------------------------------------------------------ evidence ---
        with gr.Column(elem_classes="pv-card"):
            gr.HTML(
                ui.card_head(
                    "layers",
                    "Grad-CAM evidence",
                    chip="step 03",
                ),
                container=False,
                padding=False,
            )
            gradcam_placeholder = gr.HTML(
                ui.gradcam_placeholder(), container=False, padding=False
            )
            gradcam_output = gr.ImageSlider(
                label="Original versus Grad-CAM overlay",
                show_label=False,
                height=460,
                slider_position=55,
                interactive=False,
                container=False,
                visible=False,
            )
            gr.HTML(ui.gradcam_legend(), container=False, padding=False)

        with gr.Accordion(
            "How PyroVision works, and where it should not be trusted",
            open=False,
            elem_classes="pv-accordion",
        ):
            gr.Markdown(
                f"""
**Model.** `{MODEL_NAME}`, fine-tuned on the Quebec wildfire prediction
dataset. Frames are resized to {IMG_SIZE[0]}x{IMG_SIZE[1]} and passed through
as raw `[0, 255]` pixels - EfficientNetV2's normalisation is baked into the
architecture itself.

**Grad-CAM.** The heatmap comes from gradients of the wildfire score with
respect to the last spatial feature map, so it shows the regions that pushed
the score up. Bright areas are evidence *for* the verdict, not a fire
perimeter.

**Alert line.** The decision threshold is {DECISION_THRESHOLD:.3f}, tuned to
optimise F-beta with beta=2. That deliberately favours recall: a missed
wildfire costs far more than a second look at a false alarm. A frame can
therefore be flagged while `P(wildfire)` is still under 0.5.

**Limits.** This is a research and demonstration tool trained on one public
satellite-imagery dataset. It has not been validated for operational
emergency response and must not be used for real-world safety decisions.
Smoke, cloud, sunset light and burn scars are all known confusers.
                """
            )

        gr.HTML(ui.footer(), container=False, padding=False)

    # ------------------------------------------------------------ wiring ---
    outputs = [
        verdict_output,
        meter_output,
        metrics_output,
        gradcam_output,
        gradcam_placeholder,
    ]
    analyze_btn.click(fn=predict, inputs=image_input, outputs=outputs)
    image_input.change(fn=predict, inputs=image_input, outputs=outputs)
    reset_btn.click(fn=reset, inputs=None, outputs=[image_input, *outputs])


if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=ui.pyrovision_theme(),
        css_paths=[ui.CSS_PATH],
        head=ui.HEAD,
        favicon_path=str(ui.LOGO_PATH) if ui.LOGO_PATH.exists() else None,
        # The page carries its own branded footer; Gradio's default link row
        # would sit underneath it saying the same thing twice.
        footer_links=["api"],
    )
