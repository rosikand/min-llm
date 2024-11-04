"""
File: load_datasets.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import torch
from datasets import load_dataset
import pdb 
from transformers import GPT2Tokenizer 
from torch.utils.data import DataLoader



class TinyStoriesDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling TinyStories data.
    It simply loads in the data from hugging face and interfaces with it. 
    """
    def __init__(self, split='train'):

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
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure the pad token is set
        self.max_length = 128 



    def __getitem__(self, idx):
        text = self.data[idx]

        # Tokenize and prepare the sample with padding and truncation
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {'text': text, 'input_ids': input_ids, 'attention_mask': attention_mask}

    def __len__(self):
        leng = len(self.data)
        print(f"Dataset length: {leng}")
        return leng


class TinyShakespearesDataset(torch.utils.data.Dataset):
    """
    Character-level language model. 
    A PyTorch Dataset class for handling TinyShakespeare data.
    It simply loads in the data from hugging face and interfaces with it. 
    """
    def __init__(self, split='train'):

        # load in the data 
        print("Loading TinyShakespeare dataset...")
        self.ds = load_dataset("tiny_shakespeare")
        self.train = self.preprocess_text(self.ds['train']['text'][0])
        self.validation = self.preprocess_text(self.ds['validation']['text'][0])
        self.test = self.preprocess_text(self.ds['test']['text'][0])

        if split == 'train':
            self.data = self.train
        elif split == 'validation':
            self.data = self.validation
        elif split == 'test':
            self.data = self.test
        else:
            raise ValueError(f"Invalid split: {split}. Should be 'train' or 'validation'")



    
    def preprocess_text(self, text):
        normalized_text = list(text.encode('utf-8').decode('utf-8'))


    def __getitem__(self, idx):
        text = self.ds[idx]
        
        return text 

    def __len__(self):
        return 100 

# Create a dataset instance and a data loader

def collate_fn(batch):
    """
    Collate function to dynamically pad each batch to the length of the longest sequence.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    
    # Pad input_ids and attention_masks to the max length in this batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=GPT2Tokenizer.from_pretrained("gpt2").pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {'input_ids': input_ids, 'attention_mask': attention_masks}



def test_shakespeare():
    ds = TinyStoriesDataset(split='validation')
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    sample = next(iter(dl))
    pdb.set_trace()


test_shakespeare()
