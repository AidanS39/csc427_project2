import random
import time

# Unsmoothed Unigram

# returns a dictionary of every occuring word in the corpus and their corresponding number of occurences
def compute_unigram_counts(corpus, vocab):

    # initialize unigram counts dictionary
    unigram_counts = {word: 0 for word in vocab}

    #iterate through corpus, increment word's count every time it is found in the corpus
    for i in range(len(corpus) - 1):
        unigram_counts[corpus[i]] = unigram_counts[corpus[i]] + 1
        
    # return dictionary of unigram counts
    return unigram_counts

# returns a dictionary of every occuring word in the corpus and their corresponding probability
def compute_unigram_model(corpus):

    # initialize vocab
    vocab = set(corpus)

    # calculate length of corpus
    corpus_length = len(corpus)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
   
    # return a dictionary of unigram probs 
    return {word: unigram_counts[word] / corpus_length for word in vocab}

# Unsmoothed Bigram 

# returns a dictionary of every occuring pair in the corpus and their corresponding number of occurences
def compute_bigram_counts(corpus):

    # initialize bigram counts dictionary
    bigram_counts = dict()

    #iterate through corpus, increment pair's count every time it is found in the corpus
    for i in range(len(corpus) - 1):
        # if found bigram pair is not in counts dict, add it to dictionary with a count of 1
        if (corpus[i], corpus[i+1]) not in bigram_counts:
            bigram_counts[(corpus[i], corpus[i+1])] = 1
        # if found bigram pair is in counts dict, just increment its count by 1
        else:
            bigram_counts[(corpus[i], corpus[i+1])] += 1
    
    # return dictionary of bigram counts
    return bigram_counts

# returns a dictionary of occuring pairs in the corpus and their corresponding bigram probabilitiy
def compute_bigram_model(corpus):
    # initialize the vocab
    vocab = set(corpus)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)

    # find the counts of every occuring pair in the corpus
    bigram_counts = compute_bigram_counts(corpus)

    # initialize bigram model
    bigram_model = dict()

    # iterate through bigram counts, calculate bigram probability for every occuring pair
    for pair in bigram_counts:
        # note: bigram prob = count of pair / count of first word in pair
        bigram_model[pair] = bigram_counts[pair] / unigram_counts[pair[0]]

    # return dictionary of bigram probs
    return bigram_model

## note: for unigrams, pass a unigram model as model and a unigram word as token
##       for bigrams, pass a bigram model as model and a bigram pair as token
##       if not passed correctly, incorrect value may return or error may occur.
# returns the probability of an unsmoothed unigram word / bigram pair 
def find_unsmoothed_probability(model, token):
    if token in model:
        return model[token]
    else:
        return 0

# Smoothed Unigram

def compute_smoothed_unigram_model(corpus):
    
    # initialize vocab
    vocab = set(corpus)

    # calculate length of corpus
    corpus_length = len(corpus)

    # calculate cardinality of vocab
    vocab_length = len(vocab)

    # calculate length of corpus + cardinality of vocab (to be used as denominator in unigram probability equation))
    denominator = corpus_length + vocab_length

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    # return a dictionary of smoothed unigram probs
    return {word: (unigram_counts[word] + 1) / denominator for word in vocab}

# Smoothed Bigram

def compute_smoothed_bigram_model(corpus):
    
    # initialize vocab
    vocab = set(corpus)

    # calculate length of corpus
    corpus_length = len(corpus)

    # calculate cardinality of vocab
    vocab_length = len(vocab)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)

    # find the counts of every occuring pair in the corpus
    bigram_counts = compute_bigram_counts(corpus)

    # initialize bigram model
    bigram_model = dict()

    # iterate through bigram counts, calculate smoothed bigram probability for every occuring pair
    for pair in bigram_counts:
        # note: smoothed bigram prob = (count of pair + 1) / (count of first word in pair + vocab cardinality)
        bigram_model[pair] = (bigram_counts[pair] + 1) / (unigram_counts[pair[0]] + vocab_length)

    # return dictionary of smoothed bigram probs
    return bigram_model

## note: for unigrams, pass a unigram model as model and a unigram word as token
##       for bigrams, pass a bigram model as model and a bigram pair as token
##       if not passed correctly, incorrect value may return or error may occur.
# returns the probability of a smoothed unigram word / bigram pair
def find_smoothed_probability(model, token, unigram_counts, vocab_cardinality):
    # if token is in model, return prob of token from model
    if token in model:
        return model[token]
    # if token is not in model, then compute and return smoothed probability 
    # note: if function is correctly called and token is not in the model, then it must be a bigram. unigrams will always be in the model
    else:
        return 1 / (unigram_counts[token[0]] + vocab_cardinality)

# Generate Unigram Sentence

def generate_sentence_unigram(model):
    # TODO: write function to generate sentence using a unigram model!
    return

# Generate Bigram Sentence

def generate_sentence_bigram(model):
    sentence = ["<s>"]
    index = 1
    while (sentence[index-1] != "</s>"):
        random_prob = random.random()
        print(random_prob)
        sum_prob = 0
        for bigram_pair in model:
            if (bigram_pair[0] == sentence[index-1] and model.get(bigram_pair) != 0.0):
                print(bigram_pair)
                sum_prob += model.get(bigram_pair)
                if sum_prob >= random_prob:
                    sentence.append(bigram_pair[1])
                    index += 1
                    break
        print(sentence)
        print(index)

def generate_sentence_bigram_v2(model, corpus):

    #initialize vocab
    vocab = set(corpus)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    sentence = ["<s>"]
    index = 1
    while (sentence[index-1] != "</s>"):
        random_prob = random.random()
        print(random_prob)
        sum_prob = 0
        for bigram_pair in model:
            if (bigram_pair[0] == sentence[index-1] and find_unsmoothed_prob(model, bigram_pair) != 0):
                print(bigram_pair)
                sum_prob += find_unsmoothed_prob(model, bigram_pair)
                if sum_prob >= random_prob:
                    sentence.append(bigram_pair[1])
                    index += 1
                    break
        print(sentence)
        print(index)

def generate_sentence_smoothedbigram_v2(model, corpus):

    #initialize vocab
    vocab = set(corpus)

    # find cardinality of vocab
    vocab_length = len(vocab)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    sentence = ["<s>"]
    index = 1
    while (sentence[index-1] != "</s>"):
        random_prob = random.random()
        print(random_prob)
        sum_prob = 0
        for bigram_pair in model:
            if (bigram_pair[0] == sentence[index-1] and find_smoothed_prob(model, bigram_pair, unigram_counts, vocab_length) != 0):
                print(bigram_pair)
                sum_prob += find_smoothed_prob(model, bigram_pair)
                if sum_prob >= random_prob:
                    sentence.append(bigram_pair[1])
                    index += 1
                    break
        print(sentence)
        print(index)

# Unigram Perplexity

# computes the perplexity of a unigram model
def compute_unigram_ppl(test_set, model):
    # TODO: write function to compute the perplexity of a test set using a unigram model! (is this needed?)
    return

# Bigram Perplexity

# computes the perplexity of a bigram model
def compute_bigram_ppl(test_set, model):
    total = 1 
    for t in [1 / model[test_set[i], test_set[i+1]] for i in range(len(test_set) - 1)]:
        total *= t
    return pow(total, 1 / len(test_set))

# Examples

corpus = []

with open('bnc_corpus.txt', "r") as file:
    corpus = file.read().split()

# start_time = time.time()
# unigrams = compute_unigram_model(corpus)
# end_time = time.time()
# print(f"length of vocab: {len(set(corpus))}")
# print(f"length of unigrams: {len(unigrams)}")
# print(f"Compute time for unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# 
# start_time = time.time()
# bigrams = compute_bigram_model(corpus)
# end_time = time.time()
# print(f"length of vocab squared: {pow(len(set(corpus)), 2)}")
# print(f"length of bigrams: {len(bigrams)}")
# print(f"Compute time for bigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

vocab = set(corpus)

start_time = time.time()
smoothed_unigrams = compute_smoothed_unigram_model(corpus)
end_time = time.time()
print(f"length of vocab: {len(set(corpus))}")
print(f"length of smoothed_unigrams: {len(smoothed_unigrams)}")
print(f"Compute time for smoothed unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

start_time = time.time()
smoothed_bigrams = compute_smoothed_bigram_model(corpus)
end_time = time.time()
print(f"length of vocab squared: {pow(len(set(corpus)), 2)}")
print(f"length of smoothed_bigrams: {len(smoothed_bigrams)}")
print(f"Compute time for smoothed bigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

# start_time = time.time()
# generate_sentence_bigram(bigrams)
# end_time = time.time()
# print(f"Compute time for generating sentences: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

# start_time = time.time()
# bigram_perplexity = compute_bigram_ppl(corpus, bigrams)
# end_time = time.time()
# print(f"bigram perplexity: {bigram_perplexity}")
# print(f"Compute time for calculating bigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")












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
