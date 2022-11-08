import torch
from torch.utils.data import Dataset
from mbart_modeling import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast
import pandas as pd
import json
from PIL import Image
import os
import logging
import random
logger = logging.getLogger(__name__)

class APETrainDataset(Dataset):
    
    def __init__(self, args, path, tokenizer):
        
        self.hparams = args
        self.tokenizer = tokenizer
        file_name = "APE_train.tsv" 
        data_path = os.path.join(path,file_name)
        self.max_seq_len = 40

        self.kor_token = ["ko_KR"]
        self.en_token = ["en_XX"]
        self.ape_token = ["ape_kor"]
        data = pd.read_csv(data_path, encoding = 'utf-8', sep='\t') 



        self.dataset_lst = []
        for i in range(len(data)):
            src, mt, pe = data.iloc[i]
            self.dataset_lst.append({"src": src, "mt": mt, "pe" : pe})

    def __getitem__(self, i):
        if self.hparams.train_mode == "prompting":

            src, mt, pe = self.dataset_lst[i].values()

            src_token = self.tokenizer.tokenize(src)
            mt_token = self.tokenizer.tokenize(mt)
            pe_token = self.tokenizer.tokenize(pe)
            while True:
                idx1 = random.randint(0, len(mt_token)-1)
                idx2 = random.randint(0, len(mt_token)-1)
                if idx1 != idx2 and "▁" in mt_token[idx1] and "▁" in mt_token[idx2]:
                    break
            swch1 = mt_token[idx1]
            swch2 = mt_token[idx2]
            mt_token[idx1] = swch2
            mt_token[idx2] = swch1
            input_txt = self.ape_token + self.kor_token + src_token + [self.tokenizer.sep_token] + self.en_token + mt_token + [self.tokenizer.eos_token]
            label = self.ape_token + self.en_token + pe_token + [self.tokenizer.eos_token]
        
        else:
            src, mt, pe = self.dataset_lst[i].values()
            src_token = self.tokenizer.tokenize(src)
            mt_token = self.tokenizer.tokenize(mt)
            pe_token = self.tokenizer.tokenize(pe)

            input_txt = self.kor_token + src_token + [self.tokenizer.sep_token] + self.en_token + mt_token + [self.tokenizer.eos_token]
            label = self.en_token + pe_token + [self.tokenizer.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_txt)
        label_ids = self.tokenizer.convert_tokens_to_ids(label)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        decoder_input_ids = torch.tensor(label_ids[:-1], dtype=torch.long).contiguous()

        return {"input_ids":input_ids, "decoder_input_ids": decoder_input_ids, "labels":labels}

    def collate_fn(self, batch):
        # Collate function definition
        value_lst = [list(lst.values()) for lst in batch]
        batch = list(zip(*value_lst))
        sample = {}
        
        # max len
        input_max_len = max([x.size(0) for x in batch[0]])
        label_max_len = max([x.size(0) for x in batch[2]])
        
        input_slicing = False
        output_slicing = False

        # input ids and attention masking
        inputs = torch.ones((len(batch[0]), input_max_len), dtype=torch.long)

        for i, x in enumerate(batch[0]):
            if input_slicing:
                x = x[:input_max_len]
            inputs[i,:x.size(0)] = x

        labels = torch.ones((len(batch[2]), label_max_len), dtype=torch.long)
        for i, y in enumerate(batch[2]):
            if output_slicing:
                y = y[:label_max_len]
            labels[i,:y.size(0)] = y

        return (inputs, labels)


    def __len__(self):
        return len(self.dataset_lst)



class APEValDataset(Dataset):
    
    def __init__(self, args, path, tokenizer):
        self.hparams = args
        self.tokenizer = tokenizer
        file_name = "APE_valid.tsv" 
        data_path = os.path.join(path,file_name)
        self.max_seq_len = 40

        self.kor_token = ["ko_KR"]
        self.en_token = ["en_XX"]
        self.ape_token = ["ape_kor"]

        data = pd.read_csv(data_path, encoding = 'utf-8', sep='\t')
        data = data[:int(len(data)/2)]

        self.dataset_lst = []
        for i in range(len(data)):
            src, mt, pe = data.iloc[i]
            self.dataset_lst.append({"src": src, "mt": mt, "pe" : pe})

    def __getitem__(self, i):

        if self.hparams.train_mode == "prompting":

            src, mt, pe = self.dataset_lst[i].values()
            src_token = self.tokenizer.tokenize(src)
            mt_token = self.tokenizer.tokenize(mt)
            pe_token = self.tokenizer.tokenize(pe)


            while True:
                idx1 = random.randint(0, len(mt_token)-1)
                idx2 = random.randint(0, len(mt_token)-1)
                if idx1 != idx2 and "▁" in mt_token[idx1] and "▁" in mt_token[idx2]:
                    break
            swch1 = mt_token[idx1]
            swch2 = mt_token[idx2]
            mt_token[idx1] = swch2
            mt_token[idx2] = swch1

            input_txt = self.ape_token + self.kor_token + src_token + [self.tokenizer.sep_token] + self.en_token + mt_token + [self.tokenizer.eos_token]
            label = self.ape_token + self.en_token + pe_token + [self.tokenizer.eos_token]
        
        else:
            src, mt, pe = self.dataset_lst[i].values()
            src_token = self.tokenizer.tokenize(src)
            mt_token = self.tokenizer.tokenize(mt)
            pe_token = self.tokenizer.tokenize(pe)

            input_txt = self.kor_token + src_token + [self.tokenizer.sep_token] + self.en_token + mt_token + [self.tokenizer.eos_token]
            label = self.en_token + pe_token + [self.tokenizer.eos_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_txt)
        label_ids = self.tokenizer.convert_tokens_to_ids(label)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        decoder_input_ids = torch.tensor(label_ids[:-1], dtype=torch.long).contiguous()

        return {"input_ids":input_ids, "decoder_input_ids": decoder_input_ids, "labels":labels}

    def collate_fn(self, batch):
                # Collate function definition
        value_lst = [list(lst.values()) for lst in batch]
        batch = list(zip(*value_lst))
        sample = {}
        
        # max len
        input_max_len = max([x.size(0) for x in batch[0]])
        label_max_len = max([x.size(0) for x in batch[2]])
        
        input_slicing = False
        output_slicing = False
        
        # if self.max_seq_len < input_max_len:
        #     input_max_len = self.max_seq_len
        #     input_slicing = True
        # elif self.max_seq_len < label_max_len:
        #     label_max_len = self.max_seq_len
        #     output_slicing = True
        # else:
        #     pass

        # input ids and attention masking
        inputs = torch.ones((len(batch[0]), input_max_len), dtype=torch.long)

        for i, x in enumerate(batch[0]):
            if input_slicing:
                x = x[:input_max_len]
            inputs[i,:x.size(0)] = x

        labels = torch.ones((len(batch[2]), label_max_len), dtype=torch.long)
        for i, y in enumerate(batch[2]):
            if output_slicing:
                y = y[:label_max_len]
            labels[i,:y.size(0)] = y

        return (inputs, labels)
        

    def __len__(self):
        return len(self.dataset_lst)

class APETestDataset(Dataset):
    
    def __init__(self, args, path, tokenizer):
        
        self.hparams = args
        self.tokenizer = tokenizer
        file_name = "APE_valid.tsv" 
        data_path = os.path.join(path,file_name)


        self.kor_token = ["ko_KR"]
        self.en_token = ["en_XX"]
        self.ape_token = ["ape_kor"]

        data = pd.read_csv(data_path, encoding = 'utf-8', sep='\t')
        data = data[:int(len(data)/2)]

        self.dataset_lst = []
        for i in range(len(data)):
            src, mt, pe = data.iloc[i]
            self.dataset_lst.append({"src": src, "mt": mt, "pe" : pe})

    def __getitem__(self, i):

        if self.hparams.train_mode == "prompting":

            src, mt, pe = self.dataset_lst[i].values()
            src_token = self.tokenizer.tokenize(src)
            mt_token = self.tokenizer.tokenize(mt)
            pe_token = self.tokenizer.tokenize(pe)


            while True:
                idx1 = random.randint(0, len(mt_token)-1)
                idx2 = random.randint(0, len(mt_token)-1)
                if idx1 != idx2 and "▁" in mt_token[idx1] and "▁" in mt_token[idx2]:
                    break
            swch1 = mt_token[idx1]
            swch2 = mt_token[idx2]
            mt_token[idx1] = swch2
            mt_token[idx2] = swch1

            input_txt = self.ape_token + self.kor_token + src_token + [self.tokenizer.sep_token] + self.en_token + mt_token + [self.tokenizer.eos_token]
            label = self.ape_token + self.en_token + pe_token + [self.tokenizer.eos_token]
        
        else:
            src, mt, pe = self.dataset_lst[i].values()
            src_token = self.tokenizer.tokenize(src)
            mt_token = self.tokenizer.tokenize(mt)
            pe_token = self.tokenizer.tokenize(pe)

            input_txt = self.kor_token + src_token + [self.tokenizer.sep_token] + self.en_token + mt_token + [self.tokenizer.eos_token]
            label = self.en_token + pe_token + [self.tokenizer.eos_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_txt)
        label_ids = self.tokenizer.convert_tokens_to_ids(label)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        decoder_input_ids = torch.tensor(label_ids[:-1], dtype=torch.long).contiguous()

        return {"input_ids":input_ids, "decoder_input_ids": decoder_input_ids, "labels":labels}

        
    def __len__(self):
        return len(self.dataset_lst)