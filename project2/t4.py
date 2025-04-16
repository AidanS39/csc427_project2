import math

# computes the bigram perplexity of a test set
def compute_bigram_ppl(test_set, model):
    return pow(math.prod([model[test_set[i]][test_set[i-1]] for i in range(1, len(test_set)]), len(test_set))
