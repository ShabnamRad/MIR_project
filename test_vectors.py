from indexing import *
import utils


def get_test_vectors():
    tags, docs = utils.parse_tagged_csv('phase2_test.csv')
    print("#")
    index = Index("english")
    index.build_with_docs(docs)
    index.save_to_file("index-test.pkl")
    print("#####")
    train_index = Index("english")
    train_index = train_index.load_from_file("index-train.pkl")
    terms = train_index.index.keys()
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


tags, vectors = get_test_vectors()

with open('test_vectors.data', 'wb') as f:
    pickle.dump(vectors, f)


with open('test_tags.data', 'wb') as f:
    pickle.dump(tags, f)
