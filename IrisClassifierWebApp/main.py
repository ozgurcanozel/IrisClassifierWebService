from pathlib import Path

import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pth"  # state_dict dosyanı buraya koy
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model = IrisClassifier()
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model dosyası bulunamadı: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

app = FastAPI(title="Iris Classifier")


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20)
    sepal_width: float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width: float = Field(..., gt=0, lt=20)


@app.post("/predict")
def predict(data: IrisInput):
    x = torch.tensor(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(torch.argmax(probs))
    return {
        "prediction": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
    }


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
