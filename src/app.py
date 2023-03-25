from flask import Flask, render_template, request, jsonify
import config
import torch
import pandas as pd
import pytorch_lightning as pl
import os
import json
import requests
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import warnings
warnings.filterwarnings("ignore")

splitter = SentenceSplitter(language='en')




app = Flask(__name__)


def get_prediction(input_text,num_return_sequences):
    text =  "paraphrase: " + input_text + " </s>"

    encoding = config.TOKENIZER.encode_plus(text,pad_to_max_length=True, return_tensors="pt")

    input_ids, attention_masks = encoding["input_ids"].to(config.DEVICE), encoding["attention_mask"].to(config.DEVICE)

    outputs = config.MODEL.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )

    
    lines = [config.TOKENIZER.decode(i, skip_special_tokens=True,clean_up_tokenization_spaces=True) for i in outputs]
    return lines
 
def generate(sentence_list: list[str], num_para: int):
    paraphrased_sentence = []
    paraphrased_passage = []
    for i in (sentence_list):
        
        # r = get_response(i,num_para)
        # print("="*80)
        # print("="*80)
        # print(i)
        # print("="*80) 
        # print("="*80)
        # print("="*80)
        r = get_prediction(i,num_para)
        paraphrased_sentence.append(r)
    print(paraphrased_sentence)  
    for i in list(zip(*paraphrased_sentence)):
        paraphrased_passage.append(" ".join(list(i)))
    return paraphrased_passage


    

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form   
    input_text = data["text"]
    sentence_list = splitter.split(input_text)
  
    if data["num_return_sequences"] == "":
        num_return_sequences = 1
    
    num_return_sequences = int(data["num_return_sequences"])
    predictions = generate(sentence_list, num_return_sequences)
    # predictions = get_prediction(sentence_list, num_return_sequences)
    return render_template('predict.html', predictions=predictions, input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)