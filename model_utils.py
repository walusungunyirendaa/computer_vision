from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib

MODEL_PATH = "retinopathy_model.keras"
IMG_SIZE = 224
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
GRAD_CAM_LAYER = "out_relu" 


def load_trained_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not load '{model_path}'. Re-run the notebook's "
            "model.save() cell and confirm the file sits next to app.py."
        ) from exc


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(model: tf.keras.Model, input_array: np.ndarray) -> tuple[str, float, np.ndarray]:
    probabilities = model.predict(input_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    return CLASS_NAMES[predicted_idx], float(probabilities[predicted_idx]), probabilities


def make_gradcam_heatmap(
    model: tf.keras.Model, input_array: np.ndarray, class_idx: int,
    layer_name: str = GRAD_CAM_LAYER,
) -> np.ndarray:
    backbone = model.get_layer(index=0)
    try:
        conv_layer = backbone.get_layer(layer_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Layer '{layer_name}' not found on the backbone. Run "
            "`[l.name for l in model.layers[0].layers if 'relu' in l.name]` "
            "in the notebook to find the correct name."
        ) from exc

    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input, outputs=[conv_layer.output, backbone.output]
    )

    with tf.GradientTape() as tape:
        conv_output, x = backbone_grad_model(input_array)
        for layer in model.layers[1:]:  
            x = layer(x)
        class_score = x[:, class_idx]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError("Gradient is None: conv_layer isn't on the forward path.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(image.size, Image.BILINEAR)
    )
    colormap = matplotlib.colormaps["jet"]
    colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored).convert("RGB")
    return Image.blend(image.convert("RGB"), heatmap_img, alpha)