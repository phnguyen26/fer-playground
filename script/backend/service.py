from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True) #avoid overflow
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class Classifier:
    def __init__(self, model_path: Path) -> None:
    
        self.model_path = model_path
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("L").resize((48, 48))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = array[np.newaxis, np.newaxis, :, :]
        return array

    def predict(self, image: Image.Image) -> dict[str, object]:
        input_data = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: input_data})[0]
        probabilities = _softmax(logits)[0]
        ranked_indices = np.argsort(probabilities)[::-1]

        predictions = [
            {
                "className": CLASS_NAMES[index],
                "probability": float(probabilities[index]),
            }
            for index in ranked_indices
        ]

        top_prediction = predictions[0]
        return {
            "predictedClass": top_prediction["className"],
            "confidence": top_prediction["probability"],
            "predictions": predictions,
        }
