import pdb 
from pdb import set_trace as bp 
import time
from datasets import load_dataset



def wait_then_print(seconds):
    """Print numbers from 0 to 4 and wait between prints."""
    for i in range(5):
        print(i)
        bp()
        time.sleep(seconds)



def explore_tinystories():
    """
    Explore the TinyStories dataset using the datasets library.
    """
    ds = load_dataset("roneneldan/TinyStories")
    # breakpoint 
    bp()



# ----------------------------------------------------------------

explore_tinystories()