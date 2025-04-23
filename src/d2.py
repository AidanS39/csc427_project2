import random
import time
import math
import sys


# creates an N-Gram model
def n_gram_model(corpus: list(list()), n_gram: int, smoothing: bool):

    corpus_length = 0

    vocab = set()

    smooth_factor = 0
    if smoothing is True: 
        smooth_factor = 1

    n_gram_counts = dict()
    n_gram_probs = dict()

    for n in range(1, n_gram + 1):
        n_gram_counts[n] = dict() 
        n_gram_probs[n] = dict() 

    for sentence in corpus:
        corpus_length += len(sentence)
        vocab.update(sentence)
        for n in range(1, n_gram + 1):
            if (n <= len(sentence)):
                for i in range(len(sentence) - n + 1):
                    curr_sentence = tuple(sentence[i:(i+n)])
                    if curr_sentence not in n_gram_counts[n]:
                        n_gram_counts[n][curr_sentence] = 1
                    else:
                        n_gram_counts[n][curr_sentence] += 1


    smooth_factor = 0
    vocab_length = 0
    if smoothing is True: 
        smooth_factor = 1
        vocab_length = len(vocab)
        

    if n_gram == 1: 
        for gram in n_gram_counts[n_gram]:
            n_gram_probs[n_gram][gram] = (n_gram_counts[n_gram][gram] + smooth_factor) / (corpus_length + vocab_length)
    else:
        for gram in n_gram_counts[n]:
            n_gram_probs[n_gram][gram] = (n_gram_counts[n_gram][gram] + smooth_factor) / (n_gram_counts[n_gram-1][gram[0:n_gram-1]] + vocab_length)

    return n_gram_probs[n_gram]


# generates a sentence based off an unsmoothed/smoothed unigram model
def generate_sentence_unigram(model):
    sentence = ["<s>"]
    
    while (sentence[-1] != "</s>"):
        random_prob = random.random()
        sum_prob = 0
        for unigram in model:
    
            sum_prob += model[unigram]
            if (sum_prob >= random_prob) and (unigram != "<s>"):
                sentence.append(unigram)
                break
    
        if sum_prob < random_prob:
            sentence.append("</s>")
                    
    print(sentence)

    return sentence



# generates a sentence based off an unsmoothed bigram model
def generate_sentence_unsmoothed_bigram(model, corpus):

    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    sentence = ["<s>"]
    while (sentence[-1] != "</s>"):
        random_prob = random.random()
        sum_prob = 0

        for word in vocab:
            bigram_pair = (sentence[-1], word)
            current_prob = get_unsmoothed_probability(model, bigram_pair)
            if current_prob > 0:
                sum_prob += current_prob
                if (sum_prob >= random_prob) and (bigram_pair != "<s>"):
                    sentence.append(word)
                    break
        
        if sum_prob < random_prob:
            sentence.append("</s>")
        
    print(sentence)

    return sentence



# generates a sentence based off a smoothed bigram model
def generate_sentence_smoothed_bigram(model, corpus):

    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    # calculate the cardinality of the vocab
    vocab_length = len(vocab)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    sentence = ["<s>"]
    while (sentence[-1] != "</s>"):
        random_prob = random.random()
        sum_probs = 0

        # initialize a baseline prob variable for all bigrams that dont exist in the model
        ## note: baseline_prob is the probability assigned to any non-occuring bigram.
        ##       this is because the smoothed probs of any non-occuring bigram pair are
        ##       entirely dependent on the prob of the first word in the bigram pair.
        baseline_prob = 1 / (unigram_counts[sentence[-1]] + vocab_length)

        for word in vocab:
            bigram_pair = (sentence[-1], word)
            if bigram_pair in model:
                current_prob = get_smoothed_probability(model, bigram_pair, unigram_counts, vocab_length)
            else:
                current_prob = baseline_prob
            if current_prob > 0:
                sum_probs += current_prob
                # print(sum_probs)
                if (sum_probs >= random_prob) and (bigram_pair != "<s>"):
                    sentence.append(word)
                    break
        if sum_probs < random_prob:
            sentence.append("</s>")

    print(sentence)



# computes the perplexity of a unigram model
def compute_unigram_ppl(test_set, model, unigram_counts, cardinality):
    log_total = 0

    # calculate length of test set (number of words in test set)
    test_set_length = sum([len(sentence) for sentence in test_set])

    for sentence in test_set:
        for word in sentence:
            log_total += -math.log2(get_smoothed_probability(model, word, unigram_counts, cardinality))
    return pow(2, log_total / test_set_length)



# computes the perplexity of a bigram model
def compute_bigram_ppl(test_set, model, unigram_counts, cardinality):
    log_total = 0
    
    # calculate length of test set (number of words in test set)
    test_set_length = sum([len(sentence) for sentence in test_set])
    
    #initialize bigrams from test set
    test_bigrams = list()

    for sentence in test_set:
        for i in range(len(sentence) - 1):
            test_bigrams.append((sentence[i], sentence[i+1]))

    for pair in test_bigrams:
            log_total += -math.log2(get_smoothed_probability(model, pair, unigram_counts, cardinality))

    return pow(2, log_total / test_set_length)



# Examples

if (len(sys.argv) < 2):
    print("Insufficient number of arguments (need to specify a file path!)")
    exit()
else:
    corpus_file_path = sys.argv[1]

if (len(sys.argv) < 3):
    print("Test set not specified - using default \"test_bnc_corpus.txt\"")
    test_set_path = "test_bnc_corpus.txt"
else:
    test_set_path = sys.argv[2]

corpus = []
test_set = []

with open(corpus_file_path, "r") as file:
    for sentence in file:
        tokenized_sentence = sentence.split()
        corpus.append(tokenized_sentence)

with open(test_set_path, "r") as file:
    for sentence in file:
        tokenized_sentence = sentence.split()
        test_set.append(tokenized_sentence)

print(n_gram_model(corpus, 2, True))

# initialize vocab
# vocab = set()
# for sentence in corpus:
#     vocab.update(sentence)
# 
# # find the counts of every occuring word in the corpus
# unigram_counts = compute_unigram_counts(corpus, vocab)
# 
# 
# # UNSMOOTHED UNIGRAM MODEL
# 
# start_time = time.time()
# unigrams = compute_unigram_model(corpus)
# end_time = time.time()
# print(f"Compute time for unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
#  
# 
# 
# # UNSMOOTHED BIGRAM SENTENCE GENERATION
# 
# start_time = time.time()
# generate_sentence_unsmoothed_bigram(bigrams, corpus)
# end_time = time.time()
# print(f"Compute time for generating sentences from unsmoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# 
# 
# 
# # SMOOTHED UNIGRAM SENTENCE GENERATION
# 
# start_time = time.time()
# generate_sentence_unigram(smoothed_unigrams)
# end_time = time.time()
# print(f"Compute time for generating sentences from smoothed unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# 
# 
# 
# # SMOOTHED BIGRAM SENTENCE GENERATION (NOT RELIABLE)
# 
# # start_time = time.time()
# # generate_sentence_smoothed_bigram(smoothed_bigrams, corpus)
# # end_time = time.time()
# # print(f"Compute time for generating sentences from smoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
#  
# 
# 
# # SMOOTHED UNIGRAM PERPLEXITY
# 
# start_time = time.time()
# unigram_perplexity = compute_unigram_ppl(test_set, smoothed_unigrams, unigram_counts, len(vocab))
# end_time = time.time()
# print(f"unigram perplexity: {unigram_perplexity}")
# print(f"Compute time for calculating unigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# 
# 
# 
# # SMOOTHED BIGRAM PERPLEXITY
# 
# start_time = time.time()
# bigram_perplexity = compute_bigram_ppl(test_set, smoothed_bigrams, unigram_counts, len(vocab))
# end_time = time.time()
# print(f"bigram perplexity: {bigram_perplexity}")
# print(f"Compute time for calculating bigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")



# TOP 10 MOST PROBABLE UNIGRAMS/BIGRAMS

# # calculate length of corpus
# corpus_length = sum([len(sentence) for sentence in corpus])
# 
# # find the counts of every occuring word in the corpus
# unigram_counts = compute_unigram_counts(corpus, vocab)
# 
# top_10_unigrams = list()
# top_10_unigrams.append([" ", -1])
# 
# for key in unigram_counts.keys():
#     value = unigram_counts.get(key)
#     for index in range(len(top_10_unigrams)):
#         top_value = top_10_unigrams[index][1]
#         if value > top_value:
#             top_10_unigrams.insert(index, [key, value])
#         if len(top_10_unigrams) > 10:
#             top_10_unigrams.pop()
#             break
# 
# print("Top 10 Unsmoothed Unigrams")
# for item in top_10_unigrams:
#     item[1] = item[1] / corpus_length
#     print(f"{item[0]}\t {item[1]}\t {unigrams[item[0]]}")
# 
# # find the counts of every occuring pair in the corpus
# bigram_counts = compute_bigram_counts(corpus)
# 
# top_10_bigrams = list()
# top_10_bigrams.append([(" ", " "), -1])
# 
# for key in bigram_counts.keys():
#     value = bigram_counts.get(key)
#     for index in range(len(top_10_bigrams)):
#         top_value = top_10_bigrams[index][1]
#         if value > top_value:
#             top_10_bigrams.insert(index, [key, value])
#         if len(top_10_bigrams) > 10:
#             top_10_bigrams.pop()
#             break
# 
# print("Top 10 Unsmoothed Bigrams")
# for item in top_10_bigrams:
#     item[1] = item[1] / len(bigram_counts)
#     print(f"{item[0]}\t {item[1]}\t {bigrams[item[0]]}")
# 
# 
# # TOP 10 MOST PROBABLE SMOOTHED UNIGRAMS/BIGRAMS
# 
# # calculate length of corpus
# corpus_length = sum([len(sentence) for sentence in corpus])
# 
# # find the counts of every occuring word in the corpus
# unigram_counts = compute_unigram_counts(corpus, vocab)
# 
# top_10_unigrams = list()
# top_10_unigrams.append([" ", -1])
# 
# for key in unigram_counts.keys():
#     value = unigram_counts.get(key)
#     for index in range(len(top_10_unigrams)):
#         top_value = top_10_unigrams[index][1]
#         if value > top_value:
#             top_10_unigrams.insert(index, [key, value])
#         if len(top_10_unigrams) > 10:
#             top_10_unigrams.pop()
#             break
# 
# print("Top 10 Smoothed Unigrams")
# for item in top_10_unigrams:
#     item[1] = item[1] / corpus_length
#     print(f"{item[0]}\t {item[1]}\t {smoothed_unigrams[item[0]]}")
# 
# # find the counts of every occuring pair in the corpus
# bigram_counts = compute_bigram_counts(corpus)
# 
# top_10_bigrams = list()
# top_10_bigrams.append([(" ", " "), -1])
# 
# for key in bigram_counts.keys():
#     value = bigram_counts.get(key)
#     for index in range(len(top_10_bigrams)):
#         top_value = top_10_bigrams[index][1]
#         if value > top_value:
#             top_10_bigrams.insert(index, [key, value])
#         if len(top_10_bigrams) > 10:
#             top_10_bigrams.pop()
#             break
# 
# print("Top 10 Smoothed Bigrams")
# for item in top_10_bigrams:
#     item[1] = item[1] / len(bigram_counts)
#     print(f"{item[0]}\t {item[1]}\t {smoothed_bigrams[item[0]]}")





# # NOTE: The commmented code below is used to calculate a FULL bigram model, including bigram pairs with no occurences.
# #       Due to the large amount of storage and computation it would take to create a bigram model, this is not used.
# 
# # returns a dictionary of every pair of words (bigram pair) in the vocab and their corresponding number of occurences in the corpus 
# def compute_bigram_counts(corpus, vocab):
# 
#     #initialize the bigram_counts dictionary, with each pair's count = 0
#     bigram_counts = {(w1, w2): 0 for w2 in vocab for w1 in vocab}
# 
#     #iterate through corpus, increment pair's count every time it is found in the corpus
#     for i in range(len(corpus) - 1):
#         bigram_counts[(corpus[i], corpus[i+1])] = bigram_counts[(corpus[i], corpus[i+1])] + 1;
#     
#     return bigram_counts
# 
# # returns a dictionary of every pair of words (bigram pair) in the vocab and their corresponding probability in the corpus 
# def compute_bigram_model(corpus):
#     # initialize the vocab
#     vocab = set(corpus)
# 
#     # find the counts of every word in the vocab
#     unigram_counts = compute_unigram_counts(corpus, vocab)
# 
#     # find the counts of every pair in the vocab
#     bigram_counts = compute_bigram_counts(corpus, vocab)
# 
#     # return the a dictionary of the counts
#     return {(w1, w2): bigram_counts[(w1, w2)] / unigram_counts[w1] for w2 in vocab for w1 in vocab} 
