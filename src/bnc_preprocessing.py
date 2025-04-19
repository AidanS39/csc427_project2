import nltk
from nltk.corpus.reader import bnc

reader = bnc.BNCCorpusReader(root='../corpora/bnc/Texts/', fileids=r'.*\.xml')
sents = reader.tagged_sents(strip_space=True, stem=False)

new_sents = list()

for sentence in sents:
    new_sentence = list()
    
    # add start of sentence token to new sentence
    new_sentence.append("<s>")
    
    for word in sentence:
        # if word is not labeled a punctuation mark, add to new sentences
        if word[1] != 'PUN' and word[1] != 'PUQ':
            new_sentence.append(word[0])
    
    # add end of sentence token to new sentence
    new_sentence.append("</s>")

    #add new sentence to list of new sentences
    new_sents.append(new_sentence)

# print new sentences to a txt file

space_delimeter = " "

with open("bnc_corpus.txt", "w") as corpus_file:
    for sentence in new_sents:
        corpus_file.write(space_delimeter.join(sentence) + "\n")
