# N-Grams, Language Modeling, and Perplexity
### CSC 427 (Natural Language Processing) - Project 2

## About

The N-Grams, Language Modeling, and Perplexity project encompasses the basics of N-Gram language models.
This project provides functions to create unigram and bigram language models, with both having unsmoothed and smoothed versions (specifically add-1 smoothed versions).
Also provided are multiple sentence generation functions which use a supplied language model (passed in as a parameter) to probabilistically generate sentences.
Perplexity metric functions are provided as well, which calculates the perplexity of a test set based on the language model given.

This project was developed and tested on the TCNJ ELSA HPC system using Python 3.8.6.

## Usage

### Requirements

1. **_Python 3.8.6_**: Other versions of Python may be used but are not guaranteed to work properly. Version 3.8.6 is the only version that was tested on.
2. **_bash shell_**: The project was developed on a bash-based terminal.
3. **_8 GB of memory_**: Language models inherently use heavy amounts of memory to store lots of counts and probabilities. The program was tested on a machine with 8 GB of memory. Any less memory is not guaranteed to be enough.

### Execution

1. From the project directory, navigate to the **src** directory by running the command `cd src`
2. Find the relative file paths of the corpus text file and test set text file you wish to use. Two corpora text files (bnc_corpus.txt, brown_corpus.txt) and one test set (test_bnc_corpus.txt) has been supplied, which is located in the src directory.
3. Run the command `python3 d1.py <corpus file path> <test-set file path>`, substituting `<corpus file path>` with the file path of the corpus text file, and `<test set file path>` with the file path of the test set text file.

## Project Structure

## Provided Corpora
