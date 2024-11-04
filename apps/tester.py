import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

class TinyStoriesDataset(Dataset):
    def __init__(self, split="train", max_length=1024):
        # Load the TinyStories dataset from Hugging Face
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length

        # Ensure that the GPT-2 tokenizer will pad and truncate to max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the text data from the dataset
        text = self.dataset[idx]["text"]

        # Tokenize the text and truncate/pad to max_length
        tokens = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        # The labels for language modeling are the token IDs, shifted by one
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        
        # Shift labels for language modeling to ignore padding
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def create_dataloader(batch_size=8, split="train", max_length=1024):
    dataset = TinyStoriesDataset(split=split, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage
train_loader = create_dataloader(batch_size=8, split="train")

# Check a sample batch
i = 0
for batch in train_loader:
    print(batch["input_ids"].shape)  # Should be [batch_size, max_length]
    print(batch["labels"].shape)     # Should match input_ids shape, with padding tokens set to -100
    if i > 5:
        break

    i += 1