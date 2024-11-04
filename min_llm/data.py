"""
File: data.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import torch
from datasets import load_dataset
import pdb 
from transformers import GPT2Tokenizer 
from torch.utils.data import DataLoader
import tokenizer


class TinyStoriesDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling TinyStories data.
    It simply loads in the data from hugging face and interfaces with it. 
    """
    def __init__(self, split='train', tokenizer_name="gpt2", max_length=512):

        # load in the data 
        print("Loading TinyStories dataset...")
        self.ds = load_dataset("roneneldan/TinyStories")

        if split == 'train':
            self.data = self.ds['train']['text']
        elif split == 'validation':
            self.data = self.ds['validation']['text']
        else:
            raise ValueError(f"Invalid split: {split}. Should be 'train' or 'validation'")
        

        # load the tokenizer 
        self.tokenizer = tokenizer.build_tokenizer(tokenizer_name)
        self.max_length = max_length


    def __getitem__(self, idx):
        text = self.data[idx]

        # Tokenize and prepare the sample with padding and truncation

        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True
        )

        # Extract the relevant parts
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {'text': text, 'input_ids': input_ids, 'attention_mask': attention_mask}

    def __len__(self):
        leng = len(self.data)
        print(f"Dataset length: {leng}")
        return leng
