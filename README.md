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

The Project consists of three parts: A main program `d1.py`, a preprocessed corpus, and a test set.

Both the corpus and the test set must be text files with each line serving as a sentence. Each sentence is made up of space-based tokens or words, surrounded by a start of sentence token `<s>` and an end of sentence token `</s>`. Please refer to the `bnc_corpus.txt` and `brown_corpus.txt` files as examples on the required corpus format.

The main program has various functions which serve the purpose of creating unigram and bigram language models, both with unsmoothed (Maximum Likelihood Estimate) and smoothed (with Add-1 Smoothing) versions. The bigram models that are created are considered "sparse", meaning that any bigram pair that does not exist in the corpus is not added to the bigram model.

## Provided Corpora

Provided in the project are two corpora:

1. Brown Corpus

The Brown Corpus consists of about 1.1 million tokens after preprocessing. For preprocessing we stripped standalone punctuation tokens off of the corpus. We used the NLTK library to download the Brown Corpus. The preprocessed version of the Brown Corpus is available in the repository under the text file `brown_corpus.txt`.

2. BNC Corpus: Baby Edition

The BNC Corpus: Baby Edition consists of about 4.6 million tokens after preprocessing. For preprocessing we stripped standalone punctuation tokens off of the corpus. Since this corpus was only available in XML format, we used the NLTK library to parse the sentences from the corpus, but the corpus was locally downloaded from this website: https://llds.ling-phil.ox.ac.uk/llds/xmlui/handle/20.500.14106/2553. This corpus was split into a training set and test set, which is first 80% and last 20% of the corpus, respectively. The preprocessed version of the BNC Corpus: Baby Edition is available in the repository under the training set text file `bnc_corpus.txt` and the test set text file `test_bnc_corpus.txt`.



### Corpora Citations

BNC Consortium, 2007, British National Corpus, Baby edition, Literary and Linguistic Data Service, http://hdl.handle.net/20.500.14106/2553.

Francis, W. Nelson, and Henry Kucera. Brown Corpus Manual: A Standard Corpus of Present-Day Edited American English for Use with Digital Computers. Revised and amplified ed., Department of Linguistics, Brown University, 1979. http://www.hit.uib.no/icame/brown/bcm.html

