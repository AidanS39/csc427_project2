import random
import time
import math
import sys



# returns a dictionary of every occuring word in the corpus and their corresponding number of occurences
def compute_unigram_counts(corpus, vocab):

    # initialize unigram counts dictionary
    unigram_counts = {word: 0 for word in vocab}

    #iterate through corpus, increment word's count every time it is found in the corpus
    for sentence in corpus:
        for word in sentence:
            unigram_counts[word] += 1

    # return dictionary of unigram counts
    return unigram_counts



# returns a dictionary of every occuring word in the corpus and their corresponding probability
def compute_unigram_model(corpus):

    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    # calculate length of corpus
    corpus_length = sum([len(sentence) for sentence in corpus])

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
   
    # return a dictionary of unigram probs 
    return {word: unigram_counts[word] / corpus_length for word in vocab}



# returns a dictionary of every occuring pair in the corpus and their corresponding number of occurences
def compute_bigram_counts(corpus):

    # initialize bigram counts dictionary
    bigram_counts = dict()

    #iterate through corpus, increment pair's count every time it is found in the corpus
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            # if found bigram pair is not in counts dict, add it to dictionary with a count of 1
            if (sentence[i], sentence[i+1]) not in bigram_counts:
                bigram_counts[(sentence[i], sentence[i+1])] = 1
                # if found bigram pair is in counts dict, just increment its count by 1
            else:
                bigram_counts[(sentence[i], sentence[i+1])] += 1
    
    # return dictionary of bigram counts
    return bigram_counts



# returns a dictionary of occuring pairs in the corpus and their corresponding bigram probabilitiy
def compute_bigram_model(corpus):

    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)
    
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



# returns a dictionary of occuring words in the corpus and their corresponding smoothed unigram probabilitiy
def compute_smoothed_unigram_model(corpus):
    
    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)
    
    # calculate length of corpus
    corpus_length = sum([len(sentence) for sentence in corpus])

    # calculate cardinality of vocab
    vocab_length = len(vocab)

    # calculate length of corpus + cardinality of vocab (to be used as denominator in unigram probability equation))
    denominator = corpus_length + vocab_length

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    # return a dictionary of smoothed unigram probs
    return {word: (unigram_counts[word] + 1) / denominator for word in vocab}



# returns a dictionary of occuring pairs in the corpus and their corresponding smoothed bigram probabilitiy
def compute_smoothed_bigram_model(corpus):
    
    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)
    
    # calculate length of corpus
    corpus_length = sum([len(sentence) for sentence in corpus])

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
    
    while (sentence[-1] != "</s>"):
        random_prob = random.random()
        sum_prob = 0
        for unigram in model:
    
            sum_prob += model[unigram]
            if (sum_prob >= random_prob) and (unigram != "<s>"):
                sentence.append(unigram)
                break
    
        if sum_prob < random_prob:
            sentence.append("</s>")
                    
    print(sentence)

    return sentence



# generates a sentence based off an unsmoothed bigram model
def generate_sentence_unsmoothed_bigram(model, corpus):

    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    sentence = ["<s>"]
    while (sentence[-1] != "</s>"):
        random_prob = random.random()
        sum_prob = 0

        for word in vocab:
            bigram_pair = (sentence[-1], word)
            current_prob = get_unsmoothed_probability(model, bigram_pair)
            if current_prob > 0:
                sum_prob += current_prob
                if (sum_prob >= random_prob) and (bigram_pair != "<s>"):
                    sentence.append(word)
                    break
        
        if sum_prob < random_prob:
            sentence.append("</s>")
        
    print(sentence)

    return sentence



# generates a sentence based off a smoothed bigram model
def generate_sentence_smoothed_bigram(model, corpus):
    
    count = 0
    limit = 30
    
    # initialize vocab
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)

    # calculate the cardinality of the vocab
    vocab_length = len(vocab)

    # find the counts of every occuring word in the corpus
    unigram_counts = compute_unigram_counts(corpus, vocab)
    
    sentence = ["<s>"]
    while (sentence[-1] != "</s>"):
        if count > limit:
            sentence.append("</s>")
            break
        random_prob = random.random()
        sum_probs = 0

        # initialize a baseline prob variable for all bigrams that dont exist in the model
        ## note: baseline_prob is the probability assigned to any non-occuring bigram.
        ##       this is because the smoothed probs of any non-occuring bigram pair are
        ##       entirely dependent on the prob of the first word in the bigram pair.
        baseline_prob = 1 / (unigram_counts[sentence[-1]] + vocab_length)

        for word in vocab:
            bigram_pair = (sentence[-1], word)
            if bigram_pair in model:
                current_prob = get_smoothed_probability(model, bigram_pair, unigram_counts, vocab_length)
            else:
                current_prob = baseline_prob
            if current_prob > 0:
                sum_probs += current_prob
                # print(sum_probs)
                if (sum_probs >= random_prob) and (bigram_pair != "<s>"):
                    sentence.append(word)
                    break
        count += 1
        if sum_probs < random_prob:
            sentence.append("</s>")

    print(sentence)



# computes the perplexity of a unigram model
def compute_unigram_ppl(test_set, model, unigram_counts, cardinality):
    log_total = 0

    # calculate length of test set (number of words in test set)
    test_set_length = sum([len(sentence) for sentence in test_set])

    for sentence in test_set:
        for word in sentence:
            log_total += -math.log2(get_smoothed_probability(model, word, unigram_counts, cardinality))
    return pow(2, log_total / test_set_length)



# computes the perplexity of a bigram model
def compute_bigram_ppl(test_set, model, unigram_counts, cardinality):
    log_total = 0
    
    # calculate length of test set (number of words in test set)
    test_set_length = sum([len(sentence) for sentence in test_set])
    
    #initialize bigrams from test set
    test_bigrams = list()

    for sentence in test_set:
        for i in range(len(sentence) - 1):
            test_bigrams.append((sentence[i], sentence[i+1]))

    for pair in test_bigrams:
            log_total += -math.log2(get_smoothed_probability(model, pair, unigram_counts, cardinality))

    return pow(2, log_total / test_set_length)



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
test_set = []

with open(corpus_file_path, "r") as file:
    for sentence in file:
        tokenized_sentence = sentence.split()
        corpus.append(tokenized_sentence)

with open(test_set_path, "r") as file:
    for sentence in file:
        tokenized_sentence = sentence.split()
        test_set.append(tokenized_sentence)

# initialize vocab
vocab = set()
for sentence in corpus:
    vocab.update(sentence)

# find the counts of every occuring word in the corpus
unigram_counts = compute_unigram_counts(corpus, vocab)


# UNSMOOTHED UNIGRAM MODEL

start_time = time.time()
unigrams = compute_unigram_model(corpus)
end_time = time.time()
print(f"Compute time for unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
 


# UNSMOOTHED BIGRAM MODEL

start_time = time.time()
bigrams = compute_bigram_model(corpus)
end_time = time.time()
print(f"Compute time for bigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")



# SMOOTHED UNIGRAM MODEL

start_time = time.time()
smoothed_unigrams = compute_smoothed_unigram_model(corpus)
end_time = time.time()
print(f"Compute time for smoothed unigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")



# SMOOTHED BIGRAM MODEL

start_time = time.time()
smoothed_bigrams = compute_smoothed_bigram_model(corpus)
end_time = time.time()
print(f"Compute time for smoothed bigram model: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

while True:
    #Take input
    print(f"----------------------------------------------")
    print(f"Unsmoothed Sentence Gen   0:Unigram, 1:Bigram")
    print(f"Smoothed Sentence Gen     2:Unigram, 3:Bigram")
    print(f"Smoothed Perplexity       4:Unigram, 5:Bigram")
    print(f"----------------------------------------------")
    print(f"Top 10 Probs Unsmoothed   6:Unigram & Bigram")
    print(f"Top 10 Probs Smoothed     7:Unigram & Bigram")
    print(f"----------------------------------------------")
    user_input = input("Please enter an integer or 'q' to exit: ")

    if user_input.lower() == 'q':
        print("Goodbye!")
        break

    try:
        user_input = int(user_input) #Convert to int

        if user_input == 0:
            start_time = time.time()
            generate_sentence_unigram(unigrams)
            end_time = time.time()
            print(f"Compute time for generating sentences from unsmoothed unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

        elif user_input == 1:
            start_time = time.time()
            generate_sentence_unsmoothed_bigram(bigrams, corpus)
            end_time = time.time()
            print(f"Compute time for generating sentences from unsmoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

        elif user_input == 2:
            start_time = time.time()
            generate_sentence_unigram(smoothed_unigrams)
            end_time = time.time()
            print(f"Compute time for generating sentences from smoothed unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

        elif user_input == 3:
            start_time = time.time()
            generate_sentence_smoothed_bigram(smoothed_bigrams, corpus)
            end_time = time.time()
            print(f"Compute time for generating sentences from smoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
    
        elif user_input == 4:
            start_time = time.time()
            unigram_perplexity = compute_unigram_ppl(test_set, smoothed_unigrams, unigram_counts, len(vocab))
            end_time = time.time()
            print(f"unigram perplexity: {unigram_perplexity}")
            print(f"Compute time for calculating unigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

        elif user_input == 5:
            start_time = time.time()
            bigram_perplexity = compute_bigram_ppl(test_set, smoothed_bigrams, unigram_counts, len(vocab))
            end_time = time.time()
            print(f"bigram perplexity: {bigram_perplexity}")
            print(f"Compute time for calculating bigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

        elif user_input == 6:
            # calculate length of corpus
            corpus_length = sum([len(sentence) for sentence in corpus])

            # find the counts of every occuring word in the corpus
            unigram_counts = compute_unigram_counts(corpus, vocab)

            top_10_unigrams = list()
            top_10_unigrams.append([" ", -1])

            for key in unigram_counts.keys():
                value = unigram_counts.get(key)
                for index in range(len(top_10_unigrams)):
                    top_value = top_10_unigrams[index][1]
                    if value > top_value:
                        top_10_unigrams.insert(index, [key, value])
                    if len(top_10_unigrams) > 10:
                        top_10_unigrams.pop()
                        break

            print("Top 10 Unsmoothed Unigrams")
            print(f"Word\t Count\t Probability")
            for item in top_10_unigrams:
                print(f"{item[0]}\t {item[1]}\t {unigrams[item[0]]}")

            # find the counts of every occuring pair in the corpus
            bigram_counts = compute_bigram_counts(corpus)

            top_10_bigrams = list()
            top_10_bigrams.append([(" ", " "), -1])

            for key in bigram_counts.keys():
                value = bigram_counts.get(key)
                for index in range(len(top_10_bigrams)):
                    top_value = top_10_bigrams[index][1]
                    if value > top_value:
                        top_10_bigrams.insert(index, [key, value])
                    if len(top_10_bigrams) > 10:
                        top_10_bigrams.pop()
                        break

            print("Top 10 Unsmoothed Bigrams")
            print(f"Pair\t\t Count\t Probability")
            for item in top_10_bigrams:
                print(f"{item[0]}\t {item[1]}\t {bigrams[item[0]]}")


        elif user_input == 7: 
            # calculate length of corpus
            corpus_length = sum([len(sentence) for sentence in corpus])

            # find the counts of every occuring word in the corpus
            unigram_counts = compute_unigram_counts(corpus, vocab)

            top_10_unigrams = list()
            top_10_unigrams.append([" ", -1])

            for key in unigram_counts.keys():
                value = unigram_counts.get(key)
                for index in range(len(top_10_unigrams)):
                    top_value = top_10_unigrams[index][1]
                    if value > top_value:
                        top_10_unigrams.insert(index, [key, value])
                    if len(top_10_unigrams) > 10:
                        top_10_unigrams.pop()
                        break

            print("Top 10 Smoothed Unigrams")
            print(f"Word\t Count\t Probability")
            for item in top_10_unigrams:
                print(f"{item[0]}\t {item[1]}\t {smoothed_unigrams[item[0]]}")

            # find the counts of every occuring pair in the corpus
            bigram_counts = compute_bigram_counts(corpus)

            top_10_bigrams = list()
            top_10_bigrams.append([(" ", " "), -1])

            for key in bigram_counts.keys():
                value = bigram_counts.get(key)
                for index in range(len(top_10_bigrams)):
                    top_value = top_10_bigrams[index][1]
                    if value > top_value:
                        top_10_bigrams.insert(index, [key, value])
                    if len(top_10_bigrams) > 10:
                        top_10_bigrams.pop()
                        break

            print("Top 10 Smoothed Bigrams")
            print(f"Pair\t\t Count\t Probability")
            for item in top_10_bigrams:
                print(f"{item[0]}\t {item[1]}\t {smoothed_bigrams[item[0]]}")
        else:
            print("Invalid input! Please enter a valid integer")

    except ValueError:
        print("That's not a valid option. Please enter a valid option.")
# 
# 
# 
# # UNSMOOTHED UNIGRAM SENTENCE GENERATION
# 
# start_time = time.time()
# generate_sentence_unigram(unigrams)
# end_time = time.time()
# print(f"Compute time for generating sentences from unsmoothed unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# 
# 
# 
# # UNSMOOTHED BIGRAM SENTENCE GENERATION
# 
# start_time = time.time()
# generate_sentence_unsmoothed_bigram(bigrams, corpus)
# end_time = time.time()
# print(f"Compute time for generating sentences from unsmoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
# 
# 
# 
# # SMOOTHED UNIGRAM SENTENCE GENERATION
# 
# start_time = time.time()
# generate_sentence_unigram(smoothed_unigrams)
# end_time = time.time()
# print(f"Compute time for generating sentences from smoothed unigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")

# 
# 
# # SMOOTHED BIGRAM SENTENCE GENERATION (NOT RELIABLE)
# # 
# start_time = time.time()
# generate_sentence_smoothed_bigram(smoothed_bigrams, corpus)
# end_time = time.time()
# print(f"Compute time for generating sentences from smoothed bigram: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")
 
# # 
# # 
# # # SMOOTHED UNIGRAM PERPLEXITY
# # 
# start_time = time.time()
# unigram_perplexity = compute_unigram_ppl(test_set, smoothed_unigrams, unigram_counts, len(vocab))
# end_time = time.time()
# print(f"unigram perplexity: {unigram_perplexity}")
# print(f"Compute time for calculating unigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")



# SMOOTHED BIGRAM PERPLEXITY

# start_time = time.time()
# bigram_perplexity = compute_bigram_ppl(test_set, smoothed_bigrams, unigram_counts, len(vocab))
# end_time = time.time()
# print(f"bigram perplexity: {bigram_perplexity}")
# print(f"Compute time for calculating bigram perplexity: {(int)((end_time - start_time) / 60)} minutes {(end_time - start_time) % 60} seconds")



# # TOP 10 MOST PROBABLE UNIGRAMS/BIGRAMS

# # calculate length of corpus
# corpus_length = sum([len(sentence) for sentence in corpus])

# # find the counts of every occuring word in the corpus
# unigram_counts = compute_unigram_counts(corpus, vocab)

# top_10_unigrams = list()
# top_10_unigrams.append([" ", -1])

# for key in unigram_counts.keys():
#     value = unigram_counts.get(key)
#     for index in range(len(top_10_unigrams)):
#         top_value = top_10_unigrams[index][1]
#         if value > top_value:
#             top_10_unigrams.insert(index, [key, value])
#         if len(top_10_unigrams) > 10:
#             top_10_unigrams.pop()
#             break

# print("Top 10 Unsmoothed Unigrams")
# for item in top_10_unigrams:
#     item[1] = item[1] / corpus_length
#     print(f"{item[0]}\t {item[1]}\t {unigrams[item[0]]}")

# # find the counts of every occuring pair in the corpus
# bigram_counts = compute_bigram_counts(corpus)

# top_10_bigrams = list()
# top_10_bigrams.append([(" ", " "), -1])

# for key in bigram_counts.keys():
#     value = bigram_counts.get(key)
#     for index in range(len(top_10_bigrams)):
#         top_value = top_10_bigrams[index][1]
#         if value > top_value:
#             top_10_bigrams.insert(index, [key, value])
#         if len(top_10_bigrams) > 10:
#             top_10_bigrams.pop()
#             break

# print("Top 10 Unsmoothed Bigrams")
# for item in top_10_bigrams:
#     item[1] = item[1] / len(bigram_counts)
#     print(f"{item[0]}\t {item[1]}\t {bigrams[item[0]]}")


# # TOP 10 MOST PROBABLE SMOOTHED UNIGRAMS/BIGRAMS

# # calculate length of corpus
# corpus_length = sum([len(sentence) for sentence in corpus])

# # find the counts of every occuring word in the corpus
# unigram_counts = compute_unigram_counts(corpus, vocab)

# top_10_unigrams = list()
# top_10_unigrams.append([" ", -1])

# for key in unigram_counts.keys():
#     value = unigram_counts.get(key)
#     for index in range(len(top_10_unigrams)):
#         top_value = top_10_unigrams[index][1]
#         if value > top_value:
#             top_10_unigrams.insert(index, [key, value])
#         if len(top_10_unigrams) > 10:
#             top_10_unigrams.pop()
#             break

# print("Top 10 Smoothed Unigrams")
# for item in top_10_unigrams:
#     item[1] = item[1] / corpus_length
#     print(f"{item[0]}\t {item[1]}\t {smoothed_unigrams[item[0]]}")

# # find the counts of every occuring pair in the corpus
# bigram_counts = compute_bigram_counts(corpus)

# top_10_bigrams = list()
# top_10_bigrams.append([(" ", " "), -1])

# for key in bigram_counts.keys():
#     value = bigram_counts.get(key)
#     for index in range(len(top_10_bigrams)):
#         top_value = top_10_bigrams[index][1]
#         if value > top_value:
#             top_10_bigrams.insert(index, [key, value])
#         if len(top_10_bigrams) > 10:
#             top_10_bigrams.pop()
#             break

# print("Top 10 Smoothed Bigrams")
# for item in top_10_bigrams:
#     item[1] = item[1] / len(bigram_counts)
#     print(f"{item[0]}\t {item[1]}\t {smoothed_bigrams[item[0]]}")





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
