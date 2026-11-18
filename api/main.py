"""API FastAPI para servir o modelo BERT treinado."""
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "models/bert")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    yield
    state.clear()


app = FastAPI(title="Classificador de textos de saúde (PT-BR)", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class PredictOut(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    tok, model = state["tok"], state["model"]
    enc = tok(body.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    id2label = model.config.id2label
    probs_map = {id2label[i]: float(p) for i, p in enumerate(probs)}
    best = max(probs_map, key=probs_map.get)
    return PredictOut(label=best, score=probs_map[best], probabilities=probs_map)
