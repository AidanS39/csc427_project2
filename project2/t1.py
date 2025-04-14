def Unigram_MLE(list):
    unigrams = {x : list.count(x) / len(list) for x in list}
    return(unigrams)

def Bigram_MLE(list, unigram):
    bigrams = {}
    numTokens = len(list)
    for i in range(len(list) - 1):
        if (list[i], list[i+1]) in bigrams:
            bigrams[list[i], list[i+1]] += 1
        else:
            bigrams[list[i], list[i+1]] = 1

    bigrams = {x : (bigrams[x] / (unigram[x[0]] * numTokens)) for x in bigrams}
    print(bigrams)

    
                




words = []

with open('corpus.txt', "r") as file:
    words = file.read().split()

unigrams = Unigram_MLE(words)
# print(unigrams)
Bigram_MLE(words, unigrams)