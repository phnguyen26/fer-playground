from __future__ import annotations

import io
import random
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from api.service import CLASS_NAMES, Classifier


ROOT_DIR = Path(__file__).parents[1].resolve()
print(f"ROOT_DIR: {ROOT_DIR}")
MODEL_PATH = ROOT_DIR / "api" / "vgg8.onnx"
FER_ROOT = ROOT_DIR / "data" / "FER-2013" / "test"

KAGGLE_ROOT = ROOT_DIR / "data" / "FACE-EMOTION-KAGGLE"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_sample_pool(dataset: str) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    if dataset == "fer":
        root = FER_ROOT
    else:
        root = KAGGLE_ROOT

    if not root.exists():
        return samples

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(root).as_posix()
            samples.append(
                {
                    "dataset": dataset,
                    "className": class_dir.name,
                    "imagePath": relative_path,
                    "imageUrl": f"/{dataset}-data/{relative_path}",
                    "fileName": image_path.name,
                }
            )

    return samples




classifier = Classifier(MODEL_PATH)
fer_sample_pool = build_sample_pool("fer")
kaggle_sample_pool = build_sample_pool("kaggle")

app = FastAPI(title="FER-2013 ONNX API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/fer-data", StaticFiles(directory=str(FER_ROOT)), name="fer-data")
app.mount("/kaggle-data", StaticFiles(directory=str(KAGGLE_ROOT)), name="kaggle-data")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/classes")
def get_classes() -> dict[str, list[str]]:
    return {"classes": CLASS_NAMES}


@app.get("/api/fer-samples")
def get_fer_samples() -> dict[str, object]:
    if not fer_sample_pool:
        raise HTTPException(status_code=404, detail="FER dataset not found")

    sample_count = 10
    samples = random.sample(fer_sample_pool, sample_count)
    return {"samples": samples}


@app.get("/api/kaggle-samples")
def get_kaggle_samples() -> dict[str, object]:
    if not kaggle_sample_pool:
        raise HTTPException(status_code=404, detail="Kaggle dataset not found")

    sample_count = min(10, len(kaggle_sample_pool))
    samples = random.sample(kaggle_sample_pool, sample_count)
    return {"samples": samples}


def _open_image_from_bytes(image_bytes: bytes) -> Image:
    try:
        return Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def _resolve_sample_root(dataset: str | None) -> Path:
    if dataset == "kaggle":
        return KAGGLE_ROOT
    return FER_ROOT


@app.post("/api/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    image_path: str | None = Form(default=None),
    dataset: str | None = Form(default="fer"),
) -> dict[str, object]:
    if not file and not image_path:
        raise HTTPException(status_code=400, detail="Provide file or image_path")

    if image_path:
        sample_root = _resolve_sample_root(dataset)
        root_path = (sample_root / image_path)
        with root_path.open("rb") as image_file:
            image = _open_image_from_bytes(image_file.read())
        source_name = image_path
    else:

        image = _open_image_from_bytes(await file.read())
        source_name = file.filename

    prediction = classifier.predict(image)
    return {
        "sourceName": source_name,
        **prediction,
    }
