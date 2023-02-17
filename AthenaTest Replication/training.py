# importing packages
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
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
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the model")
parser.add_argument("-p", "--enpretrainInput", dest="enpretrainInput", help="English Pre Training file for the model")
parser.add_argument("-c", "--codepretrainInput", dest="codepretrainInput", help="Code Pre Training file for the model")
parser.add_argument("-o", "--modelOutputDir", dest="outPath", help="Output Directory Path for the model")
parser.add_argument("-eo", "--ENmodelOutputDir", dest="EnOutPath", help="English Output Directory Path for the model")
parser.add_argument("-po", "--premodelOutputDir", dest="PreOutPath", help="Pre Training Output Directory Path for the model")
parser.add_argument("-s", "--sourceLabel", dest="sourceLabel", help="Source Label for the train.csv file")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-v", "--valInput", dest="valInput", help="Val Input file for the model accuracy")
parser.add_argument("-e", "--epochs", dest="epochs", help="Epochs for the model")
parser.add_argument("-pe", "--preEnEpochs", dest="preEnEpochs", help="Epochs for the model english pre train")
parser.add_argument("-ce", "--preCodeEpochs", dest="preCodeEpochs", help="Epochs for the model code pre train")
parser.add_argument("-a","--externalTestFile", dest="externalTestFile", help="External Test Files for generating the UTs")
parser.add_argument("-q","--externalTestFileOutput", dest="externalTestFileOutput", help="External Test Files output for generating the UTs")

args = parser.parse_args()
trainInput = args.trainInput
enPreTrainInput = args.enpretrainInput
codePreTrainInput = args.codepretrainInput
outPath = args.outPath
EnOutPath = args.EnOutPath
PreOutPath = args.PreOutPath
sourceLabel = args.sourceLabel
testInput = args.testInput
epochsTaken = args.epochs
epochsTakenEn = args.preEnEpochs
epochsTakenCode = args.preCodeEpochs
extTestFile = args.externalTestFile
outTestFile = args.externalTestFileOutput

class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, eval_beams, freeze_encoder, freeze_embeds):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        # self.freeze_encoder = freeze_encoder
        # self.freeze_embeds_ = freeze_embeds
        self.eval_beams = eval_beams
        self.freeze_encoder = freeze_encoder
        self.freeze_embeds1 = freeze_embeds

        if self.freeze_encoder:
            freeze_params(self.model.get_encoder())

        if self.freeze_embeds1:
            self.freeze_embeds()

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        # Create the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss':loss}

    def validation_step(self, batch, batch_idx):

        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]

        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        self.log("val_loss",val_loss)

        return {'loss': val_loss}

    # Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, text, eval_beams, early_stopping = True, max_len = 40):
        ''' Function to generate text '''
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False

# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, num_examples = 200000000):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_examples = num_examples

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.data = pd.read_csv(self.data_file, lineterminator='\n', delimiter=",", encoding='utf-8', header=None, names=[sourceLabel,'target'],on_bad_lines='skip')[:self.num_examples]
        self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.6*len(self.data)), int(.8*len(self.data))])

    # encode the sentences using the tokenizer
    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train[sourceLabel], self.train['target'])
        self.validate = encode_sentences(self.tokenizer, self.validate[sourceLabel], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test[sourceLabel], self.test['target'])

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)
        return test_data

# Create the hparams dictionary to pass in the model
# I realise that this isn't really how this is meant to be used, but having this here reminds me that I can edit it when I need
hparams = argparse.Namespace()

hparams.freeze_encoder = True
hparams.freeze_embeds = True
hparams.eval_beams = 4

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=32, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence
      Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
      Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        encoded_dict = tokenizer(
              sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
          )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
              sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
          )
    # Shift the target ids to the right
    # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
    target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)


    batch = {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
      "labels": target_ids,
    }

    return batch


def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
    '''
    Function that noises a sentence by adding <mask> tokens
    Args: sentence - the sentence to noise
        percent_words - the percent of words to replace with <mask> tokens; the number is rounded up using math.ceil
    Returns a noised sentence
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    sentence = sentence_.copy()

    num_words = math.ceil(len(sentence) * percent_words)

    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
    sample_tokens = set(np.arange(0, np.maximum(1, len(sentence)-1)))

    words_to_noise = random.sample(sample_tokens, num_words)

    # Swap out words, but not full stops
    for pos in words_to_noise:
        if sentence[pos] != '.':
            sentence[pos] = replacement_token

    # Remove redundant spaces
    sentence = re.sub(r' {2,5}', ' ', ' '.join(sentence))

    # Combine concurrent <mask> tokens into a single token; this just does two rounds of this; more could be done
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    return sentence


# Load the model
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)

bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large")

if(epochsTakenEn!=0):
    # Load the data into the model for training
    summary_data_en = SummaryDataModule(tokenizer, enPreTrainInput,
                                     batch_size = 16, num_examples = 62400002001)

if(epochsTakenCode!=0):
    # Load the data into the model for training
    summary_data_code = SummaryDataModule(tokenizer, codePreTrainInput,
                                     batch_size = 16, num_examples = 62400002100)

if(epochsTaken!=0):
    # Load the data into the model for training
    summary_data = SummaryDataModule(tokenizer, trainInput,
                                     batch_size = 16, num_examples = 62400002100)

# Load the model from a pre-saved checkpoint; alternatively use the code below to start training from scratch
# model = LitModel.load_from_checkpoint(base_dir + "checkpoint_files_2/8_ep_140k_simple_0210.ckpt",
#                                       learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)



model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, eval_beams=4, freeze_encoder=True, freeze_embeds=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model= nn.DataParallel(model)
model.to(device)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
checkpoint = ModelCheckpoint(dirpath="checkpoint_files_3/")
if(epochsTakenEn!=0):
    trainer = pl.Trainer(gpus = 1,
                         max_epochs = int(epochsTakenEn),
                         min_epochs = 1,
                         auto_lr_find = True,
                         callbacks = [checkpoint],
                         logger=tb_logger,
                         progress_bar_refresh_rate = 50)


    # Pre training english Fit the instantiated model to the data
    trainer.fit(model, summary_data_en)

    #Saving the model
    torch.save(model, EnOutPath)

if(epochsTakenCode!=0):
    trainer = pl.Trainer(gpus = 1,
                         max_epochs = int(epochsTakenCode),
                         min_epochs = 1,
                         auto_lr_find = True,
                         callbacks = [checkpoint],
                         logger=tb_logger,
                         progress_bar_refresh_rate = 50)

    # Fit the instantiated model to the data
    trainer.fit(model, summary_data_code)


    #Saving the model
    torch.save(model, PreOutPath)

if(epochsTaken!=0):
    trainer = pl.Trainer(gpus = 1,
                         max_epochs = int(epochsTaken),
                         min_epochs = 1,
                         auto_lr_find = True,
                         callbacks = [checkpoint],
                         logger=tb_logger,
                         progress_bar_refresh_rate = 50)

    # Fit the instantiated model to the data
    trainer.fit(model, summary_data)

    #saving the model

    torch.save(model, outPath)

#To Load the model
# Model class must be defined somewhere
#file path -- checkpoint.pth
# model = torch.load(outPath)
# model.eval()

def generate_code(seed_line, num_lines, model_, noise_percent = 0.25, multiple_lines = False, max_line_history = 3):
    ''' Function that generates lyrics based on previously generated lyrics
      Args: seed_line - a line to start off the machine
            num_lines - the number of lines to generate
            model_ - the model used to generate the text
            multiple_lines - whether the model generates based on multiple previous lines or just the past line
            max_line_history - the maximum number of previous lines used in the current input
      Returns a list with num_lines of rap lines
    '''
    # Put the model on eval mode
    model_.to(torch.device('cpu'))
    model_.eval()
    lyrics = []
    lyrics.append(seed_line)
    prompt_line_tokens = tokenizer(noise_sentence(seed_line, 0.2), max_length = 32, return_tensors = "pt", truncation = True)
    # Loop through the number of lines generating a new line based on the old

    line = [seed_line]
    for i in range(num_lines):
        # Print out the new line
        # print(line[0].strip())
        lyrics.append(line[0])
        line = model.generate_text(prompt_line_tokens, eval_beams = 4)
        # This deals with an artefact in the training data that I had an issue cleaning
        if line[0].find(":") != -1:
            line[0] = re.sub(r'[A-Z]+: ', '', line[0])
        # This allows the model to generate a new line conditioned on more than one line
        if multiple_lines:
            start_line = np.maximum(0, i - max_line_history)
            end_line = i
            prompt_line = ' '.join(lyrics[start_line:end_line]) # Going to end_line is fine because it is non-inclusive
        else:
            prompt_line = lyrics[i]
        prompt_line_tokens = tokenizer(noise_sentence(prompt_line, noise_percent), max_length = 32, return_tensors = "pt", truncation = True)

    return lyrics


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
    new_test = generate_code(seed_line = input, num_lines = 2, model_ = model,
                           noise_percent = 0.25, multiple_lines = False, max_line_history = 1)
    #Writing to file
    with open(outTestFile, 'a+') as f:
        f.write(new_test[2]+"\n")

testData = pd.read_csv(testInput)

references_corpus = []
candidate_corpus = []

test_cases = []

#Removing older version of the file
if os.path.exists("Test_cases.txt"):
    os.remove("Test_cases.txt")

for i in tqdm(range(len(testData)), desc="Task Completed"):
    output = [testData['target'].iloc[i].split(' ')]

    input = testData[sourceLabel].iloc[i]
    new_test = generate_code(seed_line = input, num_lines = 2, model_ = model,
                           noise_percent = 0.25, multiple_lines = False, max_line_history = 1)
    #Writing to file
    with open('Test_cases.txt', 'a+') as f:
        f.write(new_test[2]+"\n")
    input = new_test[2].split(' ')
    references_corpus.append(output)#jskanklksldklk
    candidate_corpus.append(input)
    if(i%5000==0):
        print("Intermediate accuracy :  ", bleu_score(candidate_corpus, references_corpus))
print("Final accuracy :  ", bleu_score(candidate_corpus, references_corpus))
