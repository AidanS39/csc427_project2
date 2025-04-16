def compute_unigram(corpus, word):
    return corpus.count(word) / len(corpus)

def compute_unigram_model(corpus):
    return {w : compute_unigram(corpus, w) for w in corpus}

def compute_bigram(corpus, w1, w2):
    return sum(1 for i in range(len(corpus) - 1) if corpus[i] == w1 and corpus[i+1] == w2)  / (corpus.count(w1)) 
    
def compute_bigram_model(corpus):
    return {(w1, w2): compute_bigram(corpus, w1, w2) for w2 in set(corpus) for w1 in set(corpus)} 

words = []

with open('corpus.txt', "r") as file:
    words = file.read().split()

unigrams = compute_unigram_model(words)
# print(len(words))
bigrams = compute_bigram_model(words)
print(bigrams)
