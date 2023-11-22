from transformers import RobertaTokenizer
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from transformers import PLBartTokenizer, PLBartForConditionalGeneration

import pytorch_lightning as pl
# importing packages
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import random
import re
from pathlib import Path
from argparse import ArgumentParser
import argparse
import os, json
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from pytorch_lightning import loggers as pl_loggers


parser = ArgumentParser()
parser.add_argument("-i", "--modelInput", dest="modelInput", help="Saved Model file for the testing the script")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")

args = parser.parse_args()
modelInput = args.modelInput
testInput = args.testInput
extTestFile = args.externalTestFile
outTestFile = args.externalTestFileOutput
sourceLabel = "target"

model = PLBartForConditionalGeneration.from_pretrained(modelInput)
tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-large")



def generate_output(text, tokenizer, model, max_length=1000):
    # Truncate the input text if it's too long
    if len(text) > max_length:
        text = text[:max_length]
    
    # Prepare for the model
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    
    # Generate
    outputs = model.generate(input_ids, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Generating output for the external test files
# TODO: Might change the sequence later
#Removing older version of the file
if os.path.exists(outTestFile):
    os.remove(outTestFile)
testData = None
with open(extTestFile,'r') as file:
    lines = file.readlines()
    testData = [line.rstrip() for line in lines]
for i in tqdm(range(len(testData)), desc="Task Completed: External Test Output"):
    input = testData[i]
    new_test = generate_output(input,tokenizer,model,max_length=1000)
    #Writing to file
    with open(outTestFile, 'a+') as f:
        f.write(new_test+"\n")

testData = pd.read_csv(testInput)

references_corpus = []
candidate_corpus = []

test_cases = []

#Removing older version of the file
if os.path.exists("PlBartTest_cases.txt"):
    os.remove("PlBartTest_cases.txt")

for i in tqdm(range(len(testData)), desc="Task Completed"):
    output = [testData['target'].iloc[i].split(' ')]
    input = testData[sourceLabel].iloc[i]
    new_test = generate_output(input,tokenizer,model,max_length=1000)
    # new_test = generate_code(seed_line = input, num_lines = 2, model_ = model,
    #                        noise_percent = 0.25, multiple_lines = False, max_line_history = 1)
    #Writing to file
    with open('PlBartTest_cases.txt', 'a+') as f:
        f.write(new_test+"\n")
    input = new_test.split(' ')
    references_corpus.append(output)#jskanklksldklk
    candidate_corpus.append(input)
    if(i%5000==0):
        print("Intermediate accuracy :  ", bleu_score(candidate_corpus, references_corpus))
print("Final accuracy :  ", bleu_score(candidate_corpus, references_corpus))