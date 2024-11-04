import min_llm 
from min_llm import data as md 
import pdb 
import torch 
from torch.utils.data import DataLoader
from functools import partial


dd = md.TinyStoriesDataset(split='train', max_length=512)

dl = DataLoader(dd, batch_size=4, shuffle=True, collate_fn=partial(
                    md.collate_fn, 
                    tokenizer=dd.tokenizer,
                    max_length=dd.max_length,
                    add_bos=False,
                    add_eos=True
                )
            )

sample = next(iter(dl))


pdb.set_trace() 
