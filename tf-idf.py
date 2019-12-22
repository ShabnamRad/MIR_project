from utils import parse_tagged_csv
from indexing import Index
import pickle

def get_tags_tf_idf():
    tags, docs = parse_tagged_csv('phase2_train.csv')
    print("#")
    index = Index("english")
    index.build_with_docs(docs)
    index.save_to_file("index-train.pkl")
    print("#####")
    terms = index.index.keys()
    vectors = []
    i = 0
    for doc in docs:
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

    return tags, vectors


tags, vectors = get_tags_tf_idf()
print(tags)

with open('vectors.data', 'wb') as f:
    pickle.dump(vectors, f)


with open('tags.data', 'wb') as f:
    pickle.dump(tags, f)