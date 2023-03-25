import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import transformers
import os

# DATASETS
GOOGLE_DATA = "../data/google/train.tsv"
MSRP_DATA = "../data/msrp/msr_paraphrase_train.txt"
PARABANK_DATA = "../data/parabank/parabank_5m.tsv"
QUORA_DATA = "../data/quora/quora_duplicate_questions.tsv"

# MODEL PARAMS
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 3e-4
DROPOUT = 0.2
BERT_DIM = 768
NUM_CLASSES = 1
EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0.0
ADAM_EPSILON = 1e-8
WARMUP_STEPS = 0
EARLY_STOPPING = False



# Model and Tokenizer
MODEL_NAME = "t5-base"
MODEL_PATH = "../models/"
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_PATH, model_max_length=512)

# Checkpoints
DIRPATH = "../checkpoints/"
FILENAME = "best-checkpoint"



