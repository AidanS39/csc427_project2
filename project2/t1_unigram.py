import nltk
from nltk.corpus import reuters

nltk.download('reuters')

def unigram(corpus):

    token_count = {}

    for token in corpus:
        if token in token_count:
            token_count[token] += 1
        elif token not in token_count:
            token_count[token] = 1

    unigram_mle = {}
    
    for token, count in token_count.items():
        unigram_mle[token] = (count / len(reuters.words()))

    return unigram_mle 

results = unigram(reuters.words())
print(results)
print(len(reuters.words()))

print(len([word for word in reuters.words() if word == "shines"]) / len(reuters.words()))
