from utils import *
from indexing import Index
import pickle
from math import sqrt
import numpy as np

def get_tags_tf_idf():
    tags, docs = parse_csv_cluster('Data.csv')
    print("#")
    index = Index("english")
    index.build_with_docs(docs)
    index.save_to_file("index3-train.pkl")
    print("#####")
    terms = index.index.keys()
    vectors = []
    i = 0
    for doc in docs:
        vector = []
        i += 1
        norm = 0.0
        for term in terms:
            tf = index.get_tf(term, doc['id'], 'text')
            idf = index.get_idf(term, 'text')
            vector.append(tf*idf)
            norm += (tf*idf) ** 2

        norm = sqrt(norm)

        vector = np.array(vector)
        if norm > 0.0001:
            vector = vector / norm


        vectors.append(vector)
        #print(vector)
        if i % 100 == 0:
            print(i)

    return tags, vectors


tags, vectors = get_tags_tf_idf()

with open('vectors3.data', 'wb') as f:
    pickle.dump(vectors, f)


with open('ids.data', 'wb') as f:
    pickle.dump(tags, f)