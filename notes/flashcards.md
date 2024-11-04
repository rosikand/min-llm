# Flashcards on LLM's 


## Model and data parallelism techniques 

- tensor parallelism
- column parallelism
- pipeline parallelism
- data parallelism (fully-sharded data parallelism) 


## How does tensor parallelism work? 

... 


## How does pipeline parallelism work? 

... 

## How does data parallelism work? 

... 


## Are language models generative or discriminative? 

Discriminative technically speaking. 

## Do sequences across differing batches need to have the same length? What about intra-batch?

- Intra-batch: must have the same length
- Inter-batch: can have different lengths

The transformer architecture accepts sequences of variable length, but only one sequence length at a time. 

