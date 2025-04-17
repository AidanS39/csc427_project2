import random

# Unsmoothed Unigram

def compute_unigram(corpus, word):
    return corpus.count(word) / len(corpus)

def compute_unigram_model(corpus):
    return {w : compute_unigram(corpus, w) for w in corpus}

# Unsmoothed Bigram

def compute_bigram(corpus, w1, w2):
    return sum(1 for i in range(len(corpus) - 1) if corpus[i] == w1 and corpus[i+1] == w2)  / (corpus.count(w1)) 
    
def compute_bigram_model(corpus):
    return {(w1, w2): compute_bigram(corpus, w1, w2) for w2 in set(corpus) for w1 in set(corpus)} 

# Smoothed Unigram

def compute_unigram_smoothed(corpus, word):
    return (corpus.count(word) + 1) / (len(corpus) + len(set(corpus)))

def compute_unigram_smoothed_model(corpus):
    return {w : compute_unigram(corpus, w) for w in corpus}

# Smoothed Bigram

def compute_bigram_smoothed(corpus, w1, w2):
    return (sum(1 for i in range(len(corpus) - 1) if corpus[i] == w1 and corpus[i+1] == w2) + 1) / (corpus.count(w1) + len(set(corpus))) 
    
def compute_bigram_smoothed_model(corpus):
    return {(w1, w2): compute_bigram(corpus, w1, w2) for w2 in set(corpus) for w1 in set(corpus)} 

# Generate Bigram Sentence

def generate_sentence_bigram(bigram):
    sentence = ["<s>"]
    index = 1
    while (sentence[index-1] != "</s>"):
        number = random.random()
        print(number)
        count = 0
        for pair in bigram:
            if (pair[0] == sentence[index-1] and bigram.get(pair) != 0.0):
                print(pair)
                count += bigram.get(pair)
                if count >= number:
                    sentence.append(pair[1])
                    index += 1
                    break
        print(sentence)
        print(index)

# computes the bigram perplexity of a test set

def compute_bigram_ppl(test_set, model):
    total = 1 
    for t in [1 / model[test_set[i], test_set[i+1]] for i in range(len(test_set) - 1)]:
        total *= t
    return pow(total, 1 / len(test_set))

# Examples

words = []

with open('corpus.txt', "r") as file:
    words = file.read().split()

unigrams = compute_unigram_model(words)
# print(unigrams)
bigrams = compute_bigram_model(words)
# print(bigrams)
generate_sentence_bigram(bigrams)
bigram_perplexity = compute_bigram_ppl(words, bigrams)
print(bigram_perplexity)
