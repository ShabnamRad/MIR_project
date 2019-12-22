from indexing import *
import utils
import pickle
from RF import random_forrest


def get_english_docs_tags():
    english_docs = utils.parse_csv('English.csv')
    index = Index("english")
    index = index.load_from_file("index-en.pkl")
    print("#")
    train_index = Index("english")
    train_index = train_index.load_from_file("index-train.pkl")
    terms = train_index.index.keys()
    vectors = []
    i = 0
    for doc in english_docs:
        vector = []
        i += 1
        for term in terms:
            tf = index.get_tf(term, doc['id'], 'text')
            idf = index.get_idf(term, 'text')
            vector.append(tf*idf)
        vectors.append(vector)
        #print(vector)
        if i % 100 == 0:
            print(i)

    tags = random_forrest(vectors)
    print(tags)

    with open('phase1_vectors.data', 'wb') as f:
        pickle.dump(vectors, f)

    with open('phase1_tags.data', 'wb') as f:
        pickle.dump(tags, f)

    return vectors, tags


get_english_docs_tags()
