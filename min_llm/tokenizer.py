"""
file: tokenizer.py
----------------- 
Implements various tokenizer wrappers. 
"""


import abc
import tiktoken
from transformers import GPT2Tokenizer
from typing import List, Optional, Tuple, Union, Dict
import torch
import numpy as np 


class Tokenizer(abc.ABC):
    """
    class: Tokenizer
    ----------------
    Abstract class for tokenizers. 
    Child classes should implement the `encode` and `decode` methods.
    """

    @abc.abstractmethod
    def encode(self, text):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    def test_print():
        print("This is a test")


class GPT2TokenizerTiktoken(Tokenizer):
    """
    class: GPT2TokenizerTiktoken
    ----------------
    A gpt2 tokenizer wrapper that uses tiktoken. 
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def num_vocab(self):
        print(f"\nTiktoken vocab size: {self.tokenizer.n_vocab}")
        return self.tokenizer.n_vocab

    def sample_test(self):
        text = "Hello, I'm excited to build a GPT model! 🚀!?!"
        tokens = self.encode(text)
        decoded = self.decode(tokens)
        print(f"Original text: {text}")
        print(f"Tokens (ids): {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Decoded text: {decoded}")

        # differing lengths now 
        text = "Hello, I'm excited to build a GPT model! 🚀!?!"
        text_two = "Hi my name is gpt!"
        tokens = self.encode(text)
        tokens_two = self.encode(text_two)
        decoded = self.decode(tokens)
        decoded_two = self.decode(tokens_two)
        print(f"Original text: {text}")
        print(f"Tokens (ids): {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Decoded text: {decoded}")
        print(f"Original text: {text_two}")
        print(f"Tokens (ids): {tokens_two}")
        print(f"Token count: {len(tokens_two)}")
        print(f"Decoded text: {decoded_two}")




# class GPT2TokenizerTiktoken(Tokenizer):
#     """
#     class: GPT2TokenizerTiktoken
#     ----------------
#     A gpt2 tokenizer wrapper that uses tiktoken. 
#     More explicit control over special tokens
#     Consistent interface across different use cases
#     Better type hints and documentation
#     Simplified access to common properties
#     Cleaner batch processing
#     Sensible defaults for GPT-2 specifically
#     """

#     def __init__(self):
#         self.tokenizer = tiktoken.get_encoding("gpt2")
#         # Special tokens
#         self.bos_token_id = 50256  # Default GPT-2 BOS/EOS token
#         self.eos_token_id = 50256
#         self.pad_token_id = self.eos_token_id  # Common practice to use EOS as PAD
#         self.max_length = 1024  # Default max length
        
#     def encode(self, text, add_bos=False, add_eos=False):
#         """Encode text to token ids."""
#         if isinstance(text, str):
#             tokens = self.tokenizer.encode(text)
#             if add_bos:
#                 tokens = [self.bos_token_id] + tokens
#             if add_eos:
#                 tokens = tokens + [self.eos_token_id]
#             return tokens
#         elif isinstance(text, list):
#             return [self.encode(t, add_bos, add_eos) for t in text]
#         else:
#             raise ValueError(f"Unsupported input type: {type(text)}")

#     def decode(self, tokens):
#         """Decode token ids to text."""
#         if isinstance(tokens, list) and not isinstance(tokens[0], list):
#             return self.tokenizer.decode(tokens)
#         return [self.tokenizer.decode(t) for t in tokens]
    
#     @property
#     def vocab_size(self):
#         """Get vocabulary size."""
#         return self.tokenizer.n_vocab
    
#     def pad_sequence(self, token_ids, max_length=None, pad_right=True):
#         """Pad sequence to max_length."""
#         if max_length is None:
#             max_length = self.max_length
            
#         if len(token_ids) > max_length:
#             return token_ids[:max_length]
            
#         pad_length = max_length - len(token_ids)
#         padding = [self.pad_token_id] * pad_length
        
#         if pad_right:
#             return token_ids + padding
#         return padding + token_ids
    
#     def create_attention_mask(self, token_ids):
#         """Create attention mask for padded sequence."""
#         return [1 if token != self.pad_token_id else 0 for token in token_ids]
    
#     def batch_encode(self, texts, max_length=None, add_bos=False, add_eos=False, 
#                     pad=True, return_tensors=None):
#         """
#         Batch encode texts with padding and attention masks.
#         Similar to HuggingFace's __call__ method.
#         """
#         import torch
        
#         # Encode all texts
#         encoded = self.encode(texts, add_bos=add_bos, add_eos=add_eos)
        
#         # Find max length in batch if not specified
#         if max_length is None:
#             max_length = min(max(len(ids) for ids in encoded), self.max_length)
            
#         if pad:
#             # Pad sequences and create attention masks
#             padded = [self.pad_sequence(ids, max_length) for ids in encoded]
#             attention_mask = [self.create_attention_mask(ids) for ids in padded]
            
#             if return_tensors == "pt":
#                 return {
#                     "input_ids": torch.tensor(padded),
#                     "attention_mask": torch.tensor(attention_mask)
#                 }
#             return {
#                 "input_ids": padded,
#                 "attention_mask": attention_mask
#             }
        
#         if return_tensors == "pt":
#             return {"input_ids": torch.tensor(encoded)}
#         return {"input_ids": encoded}
    
#     def __call__(self, texts, max_length=None, add_bos=False, add_eos=False,
#                  pad=True, return_tensors=None):
#         """Make the class callable like HuggingFace tokenizers."""
#         return self.batch_encode(texts, max_length, add_bos, add_eos, 
#                                pad, return_tensors)

#     def get_token_offsets(self, text: str, tokens: Optional[List[int]] = None) -> Tuple[List[str], List[int]]:
#         """Get the character offsets of tokens in the text."""
#         if tokens is None:
#             tokens = self.encode(text)
            
#         token_bytes = self.tokenizer.decode_tokens_bytes(tokens)
#         text_len, offsets = 0, []
        
#         for token in token_bytes:
#             offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
#             text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
            
#         substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
#         return substrs, offsets


class GPT2TokenizerHuggingFace:
    """
    class: GPT2TokenizerHuggingFace
    ----------------
    A gpt2 tokenizer wrapper that uses HuggingFace's tokenizer.
    """

    def __init__(self, max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set padding token to EOS token (common practice for GPT-2)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length 

    def encode(self, text: Union[str, List[str]], add_bos: bool = False, add_eos: bool = False) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token ids.
        Example: 
        tok.encode("wow", add_bos=True, add_eos=True)
        >>> [50256, 42773, 50256]  # [BOS, wow, EOS]
        """
        if isinstance(text, str):
            # Get base tokens without special tokens
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,  # No special tokens here
                return_tensors=None
            )
            # Add BOS/EOS around the main tokens
            return [self.bos_token_id] * add_bos + tokens + [self.eos_token_id] * add_eos
        elif isinstance(text, list):
            return [self.encode(t, add_bos, add_eos) for t in text]
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")

    def decode(self, tokens: Union[List[int], List[List[int]], torch.Tensor]) -> Union[str, List[str]]:
        """Decode token ids to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        if isinstance(tokens[0], list) or isinstance(tokens[0], torch.Tensor) or isinstance(tokens[0], np.ndarray):
            return [self.tokenizer.decode(t) for t in tokens]
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def bos_token_id(self) -> int:
        """Get BOS token id."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """Get EOS token id."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Get PAD token id."""
        return self.tokenizer.pad_token_id

    def create_attention_mask(self, token_ids: List[int]) -> List[int]:
        """Create attention mask for padded sequence."""
        return [1 if token != self.pad_token_id else 0 for token in token_ids]

    def batch_encode(self, 
                    texts: Union[str, List[str]], 
                    max_length: Optional[int] = None, 
                    add_bos: bool = False, 
                    add_eos: bool = False,
                    pad: bool = True,
                    return_tensors: Optional[str] = None) -> Dict:
        """Batch encode texts with padding and attention masks."""
        if max_length is None:
            max_length = self.max_length

        # Use HuggingFace's built-in batch processing
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,  # We'll handle special tokens manually
            padding=pad,
            truncation=True,
            max_length=max_length - (add_bos + add_eos),  # Reserve space for special tokens
            return_tensors=return_tensors
        )

        # Convert to lists if tensors
        input_ids = encoded['input_ids'].tolist() if return_tensors == "pt" else encoded['input_ids']
        attention_mask = encoded['attention_mask'].tolist() if return_tensors == "pt" else encoded['attention_mask']

        # Add BOS/EOS tokens if requested
        if add_bos or add_eos:
            for i in range(len(input_ids)):
                if add_bos:
                    input_ids[i] = [self.bos_token_id] + input_ids[i]
                    attention_mask[i] = [1] + attention_mask[i]
                if add_eos:
                    input_ids[i] = input_ids[i] + [self.eos_token_id]
                    attention_mask[i] = attention_mask[i] + [1]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask)
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def __call__(self, 
                 texts: Union[str, List[str]], 
                 max_length: Optional[int] = None,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 pad: bool = True,
                 return_tensors: Optional[str] = None) -> Dict:
        """Make the class callable like HuggingFace tokenizers."""
        return self.batch_encode(texts, max_length, add_bos, add_eos, pad, return_tensors)




class GPT2TokenizerHuggingFaceSimple:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad_token to eos_token

    # Delegate attribute access to the tokenizer instance to ensure all other functionality is retained
    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)




def build_tokenizer(name: str) -> Tokenizer:
    """
    Function: build_tokenizer
    ------------------------
    Factory functon for building a tokenizer based on the name. 
    """
    if name == "gpt2":
        # defaults to hf's version 
        return GPT2TokenizerHuggingFace()
    elif name == "gpt2-tiktoken":
        # return GPT2TokenizerTiktoken()
        raise NotImplementedError(f"No {name} tokenizer type.")
    elif name == "gpt2-huggingface" or name == "gpt2-hf":
        return GPT2TokenizerHuggingFace()
    else:
        raise NotImplementedError(f"No {name} tokenizer type.")
    