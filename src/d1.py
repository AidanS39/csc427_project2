import random
import time
import math
import sys


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
def get_unsmoothed_probability(model, token):
    if token in model:
        return model[token]
    else:
        return 0



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
def get_smoothed_probability(model, token, unigram_counts, vocab_cardinality):
    # if token is in model, return prob of token from model
    if token in model:
        return model[token]
    # if token is not in model, then compute and return smoothed probability 
    elif token[0] in unigram_counts:
        return 1 / (unigram_counts[token[0]] + vocab_cardinality)
    # if first word not in unigram_counts, calculate a minimum smoothed probability
    else:
        return 1 / vocab_cardinality



# generates a sentence based off an unsmoothed/smoothed unigram model
def generate_sentence_unigram(model):
    sentence = ["<s>"]
    index = 1
    
    while (sentence[index-1] != "</s>"):
        random_prob = random.random()
        print(random_prob)
        sum_prob = 0
        for unigram in model:
                sum_prob += model[unigram]
                if sum_prob >= random_prob:
                    sentence.append(unigram)
                    index += 1
                    break
        
        if sum_prob < random_prob:
            sentence.append("</s>")
                    
        print(sentence)
        print(index)

    return sentence



# generates a sentence based off an unsmoothed bigram model
def generate_sentence_bigram(model, corpus):

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

        for word in vocab:
            bigram_pair = (sentence[index-1], word)
            current_prob = get_unsmoothed_probability(model, bigram_pair)
            if current_prob > 0:
                print(bigram_pair)
                sum_prob += current_prob
                if sum_prob >= random_prob:
                    sentence.append(word)
                    index += 1
                    break
        
        if sum_prob < random_prob:
            sentence.append("</s>")
        
        print(sentence)
        print(index)

    return sentence



# generates a sentence based off a smoothed bigram model
def generate_sentence_smoothed_bigram(model, corpus):

    #initialize vocab
    vocab = set(corpus)

    # calculate the cardinality of the vocab
    vocab_length = len(vocab)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    sentence = ["<s>"]
    index = 1
    while (sentence[index-1] != "</s>"):
        random_prob = random.random()
        print(random_prob)
        sum_prob = 0

        # initialize a baseline prob variable for all bigrams that dont exist in the model
        ## note: baseline_prob is the probability assigned to any non-occuring bigram.
        ##       this is because the smoothed probs of any non-occuring bigram pair are
        ##       entirely dependent on the prob of the first word in the bigram pair.
        baseline_prob = 1 / (unigram_counts[sentence[index-1]] + vocab_length)

        for word in vocab:
            bigram_pair = (sentence[index-1], word)
            if bigram_pair in model:
                current_prob = get_smoothed_probability(model, bigram_pair, unigram_counts, vocab_length)
            else:
                current_prob = baseline_prob
            if current_prob > 0:
                print(bigram_pair)
                sum_prob += current_prob
                if sum_prob >= random_prob:
                    sentence.append(word)
                    index += 1
                    break
        if sum_prob < random_prob:
            sentence.append("</s>")

        print(sentence)
        print(index)



# computes the perplexity of a unigram model
def compute_unigram_ppl(test_set, model, unigram_counts, cardinality):
    log_total = 0
    for t in test_set:
        log_total += -math.log2(get_smoothed_probability(model, t, unigram_counts, cardinality))
    return pow(2, log_total / len(test_set))



# computes the perplexity of a bigram model
def compute_bigram_ppl(test_set, model, unigram_counts, cardinality):
    log_total = 0 
    bigrams = {(test_set[i], test_set[i+1]) for i in range(len(test_set) - 1)}

    for t in bigrams:
            log_total += -math.log2(get_smoothed_probability(model, t, unigram_counts, cardinality))
    return pow(2, log_total / len(test_set))



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

with open(corpus_file_path, "r") as file:
    corpus = file.read().split()

with open(test_set_path, "r") as file:
    test_set = file.read().split()

vocab = set(corpus)
unigram_counts = compute_unigram_counts(corpus, vocab)

# start_time = time.time()
# unigrams = compute_unigram_model(corpus)
# end_time = time.time()
# print(f"length of vocab: {len(set(corpus))}")
# print(f"length of unigrams: {len(unigrams)}")
# print(f"Compute time for unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
 
start_time = time.time()
bigrams = compute_bigram_model(corpus)
end_time = time.time()
print(f"length of vocab squared: {pow(len(set(corpus)), 2)}")
print(f"length of bigrams: {len(bigrams)}")
print(f"Compute time for bigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
 
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
# generate_sentence_unigram(unigrams)
# end_time = time.time()
# print(f"Compute time for generating sentences from unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

start_time = time.time()
generate_sentence_bigram(bigrams, corpus)
end_time = time.time()
print(f"Compute time for generating sentences from bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

# start_time = time.time()
# generate_sentence_unigram(smoothed_unigrams)
# end_time = time.time()
# print(f"Compute time for generating sentences from smoothed unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

# start_time = time.time()
# generate_sentence_smoothed_bigram(smoothed_bigrams, corpus)
# end_time = time.time()
# print(f"Compute time for generating sentences from smoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
 
start_time = time.time()
unigram_perplexity = compute_unigram_ppl(test_set, smoothed_unigrams, unigram_counts, len(vocab))
end_time = time.time()
print(f"unigram perplexity: {unigram_perplexity}")
print(f"Compute time for calculating unigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

start_time = time.time()
bigram_perplexity = compute_bigram_ppl(test_set, smoothed_bigrams, unigram_counts, len(vocab))
end_time = time.time()
print(f"bigram perplexity: {bigram_perplexity}")
print(f"Compute time for calculating bigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")












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
