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
from min_llm import tokenizer
from functools import partial


class TinyStoriesDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling TinyStories data.
    It simply loads in the data from hugging face and interfaces with it. 
    """
    def __init__(self, split='train', tokenizer_name="gpt2", max_length=512):
        print("Loading TinyStories dataset...")
        self.ds = load_dataset("roneneldan/TinyStories")
        
        if split == 'train':
            self.data = self.ds['train']['text']
        elif split == 'validation':
            self.data = self.ds['validation']['text']
        else:
            raise ValueError(f"Invalid split: {split}. Should be 'train' or 'validation'")
        
        self.tokenizer = tokenizer.build_tokenizer(tokenizer_name)
        self.max_length = max_length

    def __getitem__(self, idx):
        # Just return the text, let collate_fn handle tokenization
        pdb.set_trace()
        return {'text': self.data[idx]}
        
    def __len__(self):
        leng = len(self.data)
        print(f"Dataset length: {leng}")
        return leng


def collate_fn(batch, tokenizer, max_length=512, add_bos=False, add_eos=True):
    """
    Custom collate function with more configuration options.
    """
    texts = [item['text'] for item in batch]
    
    encoded = tokenizer(
        texts,
        add_bos=add_bos,
        add_eos=add_eos,
        pad=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {
        'text': texts,
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'decoded_text': tokenizer.decode(encoded['input_ids'])
    }


def create_dataloader(dataset, batch_size=8):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(
                    collate_fn, 
                    tokenizer=dataset.tokenizer,
                    max_length=dataset.max_length,
                    add_bos=False,
                    add_eos=True
                )
            )
    
    return dl
