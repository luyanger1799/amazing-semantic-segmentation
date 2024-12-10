from dataclasses import dataclass
from typing import Optional, Tuple

import onnx
import tensorflow as tf
import tf2onnx

from builders.model_builder import builder


@dataclass
class ModelInfo:
    name: str
    input_shape: Tuple[Optional[int], int, int, int]


NUM_CLASSES = 30

MODELS = [
    ModelInfo(name="FCN-8s", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="FCN-16s", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="FCN-32s", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="UNet", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="SegNet", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="Bayesian-SegNet", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="PAN", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="PSPNet", input_shape=(None, 288, 288, 3)),
    ModelInfo(name="RefineNet", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="DenseASPP", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="DeepLabV3", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="DeepLabV3Plus", input_shape=(None, 256, 256, 3)),
    ModelInfo(name="BiSegNet", input_shape=(None, 256, 256, 3)),
]


for model_info in MODELS:
    model, _ = builder(
        NUM_CLASSES,
        (model_info.input_shape[1], model_info.input_shape[2]),
        model_info.name,
        None,
    )

    spec = (
        tf.TensorSpec(shape=model_info.input_shape, dtype=tf.float32, name="input"),
    )

    # Convert to ONNX with TensorRT-compatible options
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
    )
    # Modify Resize node
    for node in onnx_model.graph.node:
        if node.op_type == "Resize":
            for attr in node.attribute:
                if attr.name == "nearest_mode":
                    attr.s = b"round_prefer_floor"

    # Save the modified model
    model_path = f"{model_info.name.lower()}_30.onnx"
    onnx.save(onnx_model, model_path)
    print(f"Modified model saved to {model_path}")
