"""
Grad-CAM for the wildfire classifier.

Self-contained on purpose: the HF Space doesn't ship the training repo
(`wildfire_mlops`), so this re-implements the same idea described in
`model_utils.py`'s docstring from the training project - because every
backbone (EfficientNetV2, EfficientNet, ConvNeXt, ...) model was built
directly on `backbone.output` rather than by calling `backbone(inputs)` as
a nested sub-model, every backbone layer is a first-class, single-node
member of `model.layers`. That means we can auto-detect the last
spatial (4D: batch, H, W, C) layer generically, without hardcoding a
backbone-specific layer name.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib


def find_last_spatial_layer(model: tf.keras.Model) -> str:
    """Return the name of the last layer in `model` whose output is 4D
    (batch, height, width, channels) - i.e. the last conv/activation
    feature map before global pooling. Works for any backbone built the
    way described above."""
    for layer in reversed(model.layers):
        shape = getattr(layer.output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError(
        "Could not find a 4D (spatial) layer in this model - Grad-CAM "
        "needs a conv/activation feature map to target. Is this the right "
        "model file?"
    )


def make_gradcam_heatmap(
    img_batch: np.ndarray,
    model: tf.keras.Model,
    target_layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    img_batch: float32 array, shape (1, H, W, 3), UNPREPROCESSED pixel
        values in [0, 255] - this model bakes its own rescaling/
        normalization in as the first layers (see app.py), so raw pixels
        are the correct input here, same as at training time.

    Returns (heatmap, prediction) where:
        heatmap    - float32 array, shape (h, w) in [0, 1], h/w are the
                     target layer's own spatial resolution (e.g. 7x7 for
                     EfficientNetV2B0 at 224x224 input) - the caller resizes
                     it to whatever display resolution it needs.
        prediction - the model's raw sigmoid output, P(wildfire), as a
                     Python float.
    """
    if target_layer_name is None:
        target_layer_name = find_last_spatial_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output],
    )

    img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        # Single sigmoid unit -> this IS P(wildfire); no argmax/class index
        # needed like in a softmax multi-class Grad-CAM.
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError(
            f"Gradients w.r.t. layer '{target_layer_name}' were None. "
            "This usually means that layer isn't actually on the path to "
            "the output (wrong layer name), or something upstream is not "
            "differentiable."
        )

    conv_output = tf.cast(conv_output[0], tf.float32)      # (h, w, c)
    grads = tf.cast(grads[0], tf.float32)                  # (h, w, c)

    # Global-average-pool the gradients per channel -> per-channel
    # "importance" weight (this is the core Grad-CAM step).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))      # (c,)

    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)  # (h, w)
    heatmap = tf.maximum(heatmap, 0)  # ReLU - keep only positive evidence
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy(), float(predictions.numpy()[0][0])


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.45,
    colormap_name: str = "inferno",
) -> Image.Image:
    """Resize `heatmap` up to `original_image`'s native resolution and
    alpha-blend a colormapped version on top of it. Uses the image's own
    resolution (not the model's 224x224 input) so the overlay looks crisp
    even though the underlying heatmap is coarse."""
    w, h = original_image.size

    heatmap_t = tf.image.resize(
        heatmap[..., np.newaxis].astype("float32"), (h, w), method="bilinear"
    )
    heatmap_resized = heatmap_t.numpy().squeeze()  # (h, w) in [0, 1]

    colormap = matplotlib.colormaps.get_cmap(colormap_name)
    colored = colormap(heatmap_resized)[:, :, :3]  # drop alpha channel, -> (h, w, 3) in [0,1]

    base = np.asarray(original_image.convert("RGB")).astype("float32") / 255.0
    blended = (1 - alpha) * base + alpha * colored
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype("uint8"))
