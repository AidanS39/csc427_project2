def Bigram_MLE(list):
    bigrams = {}
    numTokens = len(list)
    print(list)
    print(numTokens)
    for i in range(len(list) - 1):
        if (list[i], list[i+1]) in bigrams:
            bigrams[list[i], list[i+1]] += 1
        else:
            bigrams[list[i], list[i+1]] = 1

    for item in bigrams:
        print(item, bigrams[item])
    print(dict.items(bigrams))
    print(numTokens)
    
                




words = []

with open('corpus.txt', "r") as file:
    words = file.read().split()


Bigram_MLE(words)