# Assignment 2
# NLP language model assessments

This project implements a unigram language model and a bigram language model in Python.The models 
are then trained from a tokenized corpus that uses "add-k" for smoothing to avoid zero probabilities.
The program later also evaluates model performance using perplexity inside of the main function.

# Files
- language_modeling.py 
- samiam.train
- samiam.test

# How to run
1. Make sure to be root of directory with all files above
2. Run following command in terminal (depending on python version, I used python3):

python language_modeling.py oR
python3 language_modeling.py

# Output
prints the following:
- training perplexity for unigram
- training perplexity for bigram
- test perplexity for unigram
- test perplexity for bigram