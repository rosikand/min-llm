"""
File: load_datasets.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import torch
from datasets import load_dataset
import pdb 

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


    def __getitem__(self, idx):
        text = self.data[idx]
        
        return text 

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

        pdb.set_trace()

    def __getitem__(self, idx):
        text = self.ds[idx]
        
        return text 

    def __len__(self):
        return 100 

# Create a dataset instance and a data loader

def test_shakespeare():
    ds = TinyShakespearesDataset()
    pdb.set_trace()


test_shakespeare()
