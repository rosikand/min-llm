import min_llm 
from min_llm import data as md 
import pdb 
import torch 
from torch.utils.data import DataLoader
from functools import partial


dd = md.TinyStoriesDataset(split='validation', max_length=2)

dl = md.create_dataloader(dd, batch_size=8)

sample = next(iter(dl))


pdb.set_trace() 
