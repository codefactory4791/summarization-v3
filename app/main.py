from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Body, Request
from pathlib import Path
import torch
from google.cloud import storage
import os
import pandas as pd
from torch import nn
import transformers
from transformers import AutoTokenizer
from app.model.model import predict_pipeline
from fastapi.exceptions import RequestValidationError
from app.model.model import __version__ as model_version
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


#__version__ = "1.0"

#BASE_DIR = Path(__file__).resolve(strict=True).parent

max_input_length = 1024
max_target_length = 128


app = FastAPI()


class TextIn(BaseModel):
    text: list[str]


class PredictionOut(BaseModel):
    summary: list[str]


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
	
    predicted_summary = predict_pipeline(payload.text)
    return PredictionOut(summary = predicted_summary)
