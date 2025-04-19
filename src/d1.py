import random
import time

# Unsmoothed Unigram

def compute_unigram_counts(corpus, vocab):
    unigram_counts = {word: 0 for word in vocab}
    for i in range(len(corpus) - 1):
        unigram_counts[corpus[i]] = unigram_counts[corpus[i]] + 1;
    return unigram_counts

def compute_unigram_model(corpus):
    vocab = set(corpus)
    corpus_length = len(corpus)
    unigram_counts = compute_unigram_counts(corpus, vocab)
    return {word: unigram_counts[word] / corpus_length for word in vocab}

# Unsmoothed Bigram

def compute_bigram_counts(corpus, vocab):
    bigram_counts = {(w1, w2): 0 for w2 in vocab for w1 in vocab}
    for i in range(len(corpus) - 1):
        bigram_counts[(corpus[i], corpus[i+1])] = bigram_counts[(corpus[i], corpus[i+1])] + 1;
    return bigram_counts

def compute_bigram_model(corpus):
    vocab = set(corpus)
    unigram_counts = compute_unigram_counts(corpus, vocab)
    bigram_counts = compute_bigram_counts(corpus, vocab)
    return {(w1, w2): bigram_counts[(w1, w2)] / unigram_counts[w1] for w2 in vocab for w1 in vocab} 

# Smoothed Unigram

def compute_unigram_smoothed(corpus, word):
    return (corpus.count(word) + 1) / (len(corpus) + len(set(corpus)))

def compute_unigram_smoothed_model(corpus):
    vocab = set(corpus)
    return {w : compute_unigram_smoothed(corpus, w) for w in vocab}

# Smoothed Bigram

def compute_bigram_smoothed(corpus, w1, w2, vocab_length):
    return (sum(1 for i in range(len(corpus) - 1) if corpus[i] == w1 and corpus[i+1] == w2) + 1) / (corpus.count(w1) + vocab_length) 
    
def compute_bigram_smoothed_model(corpus):
    vocab = set(corpus)
    vocab_length = len(vocab)
    return {(w1, w2): compute_bigram_smoothed(corpus, w1, w2, vocab_length) for w2 in vocab for w1 in vocab} 

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

# Unigram Perplexity

def compute_unigram_ppl(test_set, model):
    # TODO: write function to compute the perplexity of a test set using a unigram model! (is this needed?)
    return

# Bigram Perplexity

def compute_bigram_ppl(test_set, model):
    total = 1 
    for t in [1 / model[test_set[i], test_set[i+1]] for i in range(len(test_set) - 1)]:
        total *= t
    return pow(total, 1 / len(test_set))

# Examples

corpus = []

with open('bnc_corpus.txt', "r") as file:
    corpus = file.read().split()

start_time = time.time()

unigrams = compute_unigram_model(corpus)
# print(unigrams)
end_time = time.time()
print(f"Compute time for unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

bigrams = compute_bigram_model(corpus)
# print(bigrams)
end_time = time.time()
print(f"Compute time for bigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

# smoothed_unigrams = compute_unigram_smoothed_model(corpus)
# print(smoothed_unigrams)
# smoothed_bigrams = compute_bigram_smoothed_model(corpus)
# print(smoothed_bigrams)

# generate_sentence_bigram(bigrams)

end_time = time.time()
print(f"Compute time for generating sentences: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# bigram_perplexity = compute_bigram_ppl(corpus, bigrams)
# print(bigram_perplexity)
