from fastapi import FastAPI, Body, Request
from pathlib import Path
import torch
import os
import pandas as pd
from torch import nn
from google.cloud import storage
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


__version__ = "1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

max_input_length = 1024
max_target_length = 128

def download_artifacts():

    model_path = os.path.abspath("/app/app/model/model_artifacts")
    model_checkpoint ='t5-small'
    tf_model = None
    tokenizer = None

    print("Downloading Model")
    bucket_name = "summarization_bucket_2023"
    destination_file_name = f"{model_path}/pytorch_model.bin"
    storage_client = storage.Client("call-summarizatiion")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("pytorch_model.bin")
    blob.download_to_filename(destination_file_name)
    print("Model Downloaded")


    if os.path.isdir(model_path):


        print("Loading Model")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Model Loaded")
    else:
        print("No Model Artifacts Found")
        #tf_model = None

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print("Tokenizer Loaded")
    
    return tf_model,tokenizer


tf_model,tokenizer = download_artifacts()


# input is of type list of string List[str], where each item in the list is an article which needs to be summarized
def predict_pipeline(text_input):

    text = ["summarize : " + item for item in text_input]

    inputs = tokenizer(text, max_length=max_input_length, truncation=True, padding='max_length', return_tensors="pt").input_ids

    outputs = tf_model.generate(inputs, max_length=max_target_length, do_sample=False, num_beams = 3)

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = [pred.strip() for pred in predictions]

    # result = pd.DataFrame(list(zip(text_input, predictions)))
    # result.columns = ['Text_Input','Summary']
    # result.to_csv("text_summary.csv")


    return predictions
