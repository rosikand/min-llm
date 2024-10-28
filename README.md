# Implementing transformers

> This is a learning repo where I will take notes on and implement LLM's from scratch, mainly for learning purposes. Essentially, it will collate notes, resources, and code implementations for learning about LLM's.

## Code repo structure


## Goal 

Implement [llama 3](https://arxiv.org/pdf/2407.21783) in JAX and train a mini version of it. Implement all the fancy techniques discussed in the linked paper. Also make package of useful llm utilities like [here](https://github.com/rosikand/cs197-library/tree/main). 


## Resources


### Blog posts

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Note 10: Self-Attention & Transformers from CS 224N](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)
- Extra:
  - [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/#kv-cache)

### Papers

*Chronologically ordered*

- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Books

- [Build a Large Language Model (From Scratch) by Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Speech and Language Processing by Dan Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/)
- [RLHF book](https://rlhfbook.com/)


### Posts 

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)


## Why? 

- Jax has better parallelization primitives which are useful for training large models 
- Jax is lower-level and more similar to numpy, which forces you to dive deeper into the concepts
- Developing at a lower-level will make it easier to implement custom add-ons like speeding up inference with CUDA kernels or porting the inference module to C/Rust
