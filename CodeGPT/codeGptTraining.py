from transformers import RobertaTokenizer
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
# importing packages
import transformers
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler
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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from datasets import load_dataset, dataset_dict, DatasetDict


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

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java")
# prefix = "Summarize Ruby: "
# max_input_length = 256
# max_target_length = 128

def preprocess_examples(examples):
  # encode the code-docstring pairs
  max_input_length = 1024
  max_target_length = 1024
  codes = examples[sourceLabel]
  docstrings = examples['target']

  inputs = [code for code in codes]
  model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

  # encode the summaries
  labels = tokenizer(docstrings, max_length=max_target_length, padding="max_length", truncation=True).input_ids

  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)

  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs


class CodeT5(pl.LightningModule):
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000,tokenizerA, dataModule, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        self.tokenizer = tokenizerA
        self.dataModule = dataModule

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
    '''
    articles, and you should find the file medium-articles.zipin your current directory.
    Load the dataset

    Let’s import the necessary libraries.

    We load the dataset using the load_dataset function from the datasets package.
    Split the dataset into train, validation, and test set

    As a common practice, we split the dataset into:
    '''

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(self.train_dataloader())
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return self.dataModule.train_dataloader()

    def val_dataloader(self):
        return self.dataModule.val_dataloader()

    def test_dataloader(self):
        return self.dataModule.test_dataloader()
    # Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, text, eval_beams, early_stopping = True, max_len = 100):
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



# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, num_examples = 200000000):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.train_test_valid_dataset = None

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        datasetWhole = load_dataset("csv", data_files=self.data_file)
        # self.data = pd.read_csv(self.data_file, lineterminator='\n', delimiter=",", encoding='utf-8', header=None, names=[sourceLabel,'target'],on_bad_lines='skip')[:self.num_examples]
        # self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.6*len(self.data)), int(.8*len(self.data))])
        # 90% train, 10% test + validation
        train_testvalid = datasetWhole['train'].train_test_split(test_size=0.1)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        train_test_valid_dataset = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})
        self.train_test_valid_dataset = train_test_valid_dataset

    # encode the sentences using the tokenizer
    def setup(self):
        # self.train = Dataset.from_pandas(self.train)
        # self.train = self.train.map(preprocess_examples, batched=True)
        # self.validate = Dataset.from_pandas(self.validate)
        # self.validate = self.validate.map(preprocess_examples, batched=True)
        # self.test = Dataset.from_pandas(self.test)
        # self.test = self.test.map(preprocess_examples, batched=True)
        self.train_test_valid_dataset = self.rain_test_valid_dataset.map(preprocess_examples, batched=True)
        self.train_test_valid_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        # train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8)
        # valid_dataloader = DataLoader(dataset['valid'], batch_size=4)
        # test_dataloader = DataLoader(dataset['test'], batch_size=4)

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        train_data = DataLoader(self.train_test_valid_dataset['train'], shuffle=True, batch_size=8)
        return train_data

    def val_dataloader(self):
        val_data = DataLoader(self.train_test_valid_dataset['valid'], batch_size=4)
        return val_data

    def test_dataloader(self):
        test_data = DataLoader(self.train_test_valid_dataset['test'], batch_size=4)
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

def generate_output(text,tokenizer,model, max_len=256):
    # prepare for the model
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    # generate
    outputs = model.generate(input_ids, max_length = max_len)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



if __name__ == '__main__':
    if(epochsTakenEn!=0):
        # Load the data into the model for training
        summary_data_en = SummaryDataModule(tokenizer, enPreTrainInput,
                                         batch_size = 16, num_examples = 62400002001)
        summary_data_en.prepare_data()
        summary_data_en.setup()

    if(epochsTakenCode!=0):
        # Load the data into the model for training
        summary_data_code = SummaryDataModule(tokenizer, codePreTrainInput,
                                         batch_size = 16, num_examples = 62400002100)

        summary_data_code.prepare_data()
        summary_data_code.setup()
    if(epochsTaken!=0):
        # Load the data into the model for training
        summary_data = SummaryDataModule(tokenizer, trainInput,
                                         batch_size = 16, num_examples = 62400002100)
        summary_data.prepare_data()
        summary_data.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-java")
    model = nn.DataParallel(model)
    model.to(device)


    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="codeGpt_logs/")
    checkpoint = ModelCheckpoint(dirpath="checkpoint_codeGpt_3/")
    if(epochsTakenEn!=0):
        model = CodeT5(tokenizerA=tokenizer,dataModule=summary_data_en,model=model)
        # model= nn.DataParallel(model)
        # model.to(device)
        trainer = Trainer(gpus=2,
                             default_root_dir="/codeGpt/EnCheckpoints",
                             max_epochs = int(epochsTakenEn),
                             min_epochs = 1,
                             auto_lr_find = True,
                             callbacks=[early_stop_callback, lr_monitor],
                             logger=tb_logger,
                             progress_bar_refresh_rate = 50)


        # Pre training english Fit the instantiated model to the data
        trainer.fit(model)

        #Saving the model
        # save_directory "." # save in the current working directory, you can change this of course
        model.model.save_pretrained(EnOutPath)

    if(epochsTakenCode!=0):
        model = CodeT5(tokenizerA=tokenizer,dataModule=summary_data_code,model=model)
        # model= nn.DataParallel(model)
        # model.to(device)
        trainer = Trainer(gpus =2,
                             default_root_dir="/codeGpt/CodeCheckpoints",
                             max_epochs = int(epochsTakenCode),
                             min_epochs = 1,
                             auto_lr_find = True,
                             callbacks = [early_stop_callback, lr_monitor],
                             logger=tb_logger,
                             progress_bar_refresh_rate = 50)

        # Fit the instantiated model to the data
        trainer.fit(model)


        #Saving the model
        model.model.save_pretrained(PreOutPath)

    if(epochsTaken!=0):
        model = CodeT5(tokenizerA=tokenizer,dataModule=summary_data,model=model)
        # model= nn.DataParallel(model)
        # model.to(device)
        trainer = Trainer(gpus = 2,
                             default_root_dir="/codeGpt/ModelCheckpoints",
                             max_epochs = int(epochsTaken),
                             min_epochs = 1,
                             auto_lr_find = True,
                             callbacks = [early_stop_callback, lr_monitor],
                             logger=tb_logger,
                             progress_bar_refresh_rate = 50)

        # Fit the instantiated model to the data
        trainer.fit(model)

        #saving the model

        model.model.save_pretrained(outPath)


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
        new_test = generate_output(input,tokenizer,model,max_len=512)
        #Writing to file
        with open(outTestFile, 'a+') as f:
            f.write(new_test+"\n")

    testData = pd.read_csv(testInput)

    references_corpus = []
    candidate_corpus = []

    test_cases = []

    #Removing older version of the file
    if os.path.exists("codeGptTest_cases.txt"):
        os.remove("codeGptTest_cases.txt")

    for i in tqdm(range(len(testData)), desc="Task Completed"):
        output = [testData['target'].iloc[i].split(' ')]

        input = testData[sourceLabel].iloc[i]
        new_test = generate_output(input,tokenizer,model, max_len=512)
        # new_test = generate_code(seed_line = input, num_lines = 2, model_ = model,
        #                        noise_percent = 0.25, multiple_lines = False, max_line_history = 1)
        #Writing to file
        with open('codeGptTest_cases.txt', 'a+') as f:
            f.write(new_test+"\n")
        input = new_test.split(' ')
        references_corpus.append(output)#jskanklksldklk
        candidate_corpus.append(input)
        if(i%5000==0):
            print("Intermediate accuracy :  ", bleu_score(candidate_corpus, references_corpus))
    print("Final accuracy :  ", bleu_score(candidate_corpus, references_corpus))
