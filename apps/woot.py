import min_llm 
from min_llm import data as md 
import pdb 
import torch 
from torch.utils.data import DataLoader



# def collate_fn(batch):
#     pdb.set_trace() 
#     return -1 


def collate_fn(batch):
    texts = [item['text'] for item in batch]
    tokenized = md.TinyStoriesDataset.tokenizer.batch_encode(texts, pad=True, return_tensors='pt')
    return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}


dd = md.TinyStoriesDataset(split='train', max_length=512)

dl = DataLoader(dd, batch_size=4, shuffle=True, collate_fn=collate_fn)

sample = next(iter(dl))



# Create dataloader with specific settings
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=partial(
        collate_fn, 
        tokenizer=tokenizer,
        max_length=512,
        add_bos=False,
        add_eos=True
    ),
    pin_memory=True,
    drop_last=True
)