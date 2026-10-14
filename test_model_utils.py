import numpy as np
import tensorflow as tf
from PIL import Image

from model_utils import preprocess, make_gradcam_heatmap, overlay_heatmap, IMG_SIZE


def _dummy_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    features = tf.keras.layers.Conv2D(
        3, 3, padding="same", activation="relu", name="out_relu"
    )(inputs)
    backbone = tf.keras.Model(inputs, features, name="backbone")
    return tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])

def test_preprocess_shape_dtype_and_range():
    image = Image.new("RGB", (400, 300), color=(120, 60, 30))
    result = preprocess(image)
    assert result.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_preprocess_handles_grayscale_input():
    image = Image.new("L", (100, 100), color=200)
    result = preprocess(image)
    assert result.shape == (1, IMG_SIZE, IMG_SIZE, 3)  # converted to RGB


def test_gradcam_heatmap_shape_and_range():
    model = _dummy_model()
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    heatmap = make_gradcam_heatmap(model, dummy_input, class_idx=0)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_raises_on_missing_layer():
    model = _dummy_model()
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    try:
        make_gradcam_heatmap(model, dummy_input, class_idx=0, layer_name="nonexistent")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_overlay_heatmap_returns_same_size_image():
    image = Image.new("RGB", (224, 224))
    heatmap = np.random.rand(7, 7).astype("float32")
    result = overlay_heatmap(image, heatmap)
    assert result.size == image.size