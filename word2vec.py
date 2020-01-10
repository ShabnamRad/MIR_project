from gensim.models import Word2Vec
import gensim
from utils import *
import pickle
import numpy as np
import nltk
from utils import parse_csv

english_docs = parse_csv('English.csv')

persian_stop_words = []
english_stop_words = []


def find_stop_words(docs_data):
    word_count = dict()
    doc_fq = dict()
    for doc in docs_data:
        text = "%s %s" % (doc['title'], doc['text'])
        for word in get_tokens(text):
            if is_punctuation(word):
                continue
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
            if word not in doc_fq:
                doc_fq[word] = set()
            doc_fq[word].add(doc['id'])

    doc_count = len(docs_data)
    threshold = sorted(word_count.values())[-30]
    for word in word_count.keys():
        if len(doc_fq[word]) >= 0.50 * doc_count or \
                word_count[word] >= threshold:
            english_stop_words.append(word)


def get_tokens(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w.lower() for w in tokens]
    return tokens


def is_punctuation(token):
    return sum([(c in token) for c in r'\.:!،؛؟»\]\)\}«\[\(\{?*=|-;,']) > 0


def is_stop_word(token):
    return token in english_stop_words


def process(raw_text, remove_stop_words=True):
    # normalizer & tokenizer
    tokens = get_tokens(raw_text)

    final_tokens = []
    for token in tokens:
        flag = False

        # remove stop words
        if remove_stop_words:
            flag |= is_stop_word(token)

        # remove punctuations
        flag |= is_punctuation(token)

        if len(token):
            final_tokens.append(token)


    return final_tokens


find_stop_words(english_docs)



tags, docs = parse_csv_cluster('Data.csv')
processed = []
for doc in docs:
    tokens = process(doc['text'])
    processed.append(tokens)


print("**")
#model = Word2Vec(processed, min_count = 1,  size = 100, window = 5)
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print("*")
wvdata = []

i = 0
total = 0
notin = 0
for p in processed:
    if i %100 == 0:
        print(i)
    i += 1
    v = np.array([0.0 for i in range(300)])
    for word in p:
        total += 1
        if word not in model.wv.vocab:
            notin += 1
            continue

        v += model[word].copy()

    if len(p) is not 0:
        v = v / int(len(p))
    wvdata.append(v)

wvdata = np.array(wvdata)
print(total)
print(notin)
with open('word2vec.data', 'wb') as f:
    pickle.dump(wvdata, f)