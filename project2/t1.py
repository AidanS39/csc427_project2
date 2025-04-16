def Unigram_MLE(list):
    unigrams = {x : list.count(x) / len(list) for x in list}
    return(unigrams)

def Bigram_MLE(list, unigram):
    bigrams = {(list[i], list[i+1]): sum(1 for j in range(len(list) - 1) if list[j] == list[i] and list[j+1] == list[i+1]) / (unigram[list[i]] * len(list)) for i in range(len(list) - 1)}
    return bigrams

    
                




words = []

with open('corpus.txt', "r") as file:
    words = file.read().split()

unigrams = Unigram_MLE(words)
print(unigrams)
bigrams = Bigram_MLE(words, unigrams)
print(bigrams)