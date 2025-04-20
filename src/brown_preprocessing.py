import nltk
from nltk.corpus import brown
import string

nltk.download('brown')

# Get all sentences from the corpus
sents = brown.sents()

new_sents = []

for sentence in sents:
    new_sentence = ["<s>"]
    
    for word in sentence:
        # Strip punctuation 
        if word not in string.punctuation:
            new_sentence.append(word)
    
    new_sentence.append("</s>")
    new_sents.append(new_sentence)

# Save the sentences to a file
with open("brown_corpus.txt", "w") as corpus_file:
    for sentence in new_sents:
        corpus_file.write(" ".join(sentence) + "\n")
# `` = opening quote
# '' = closing quote