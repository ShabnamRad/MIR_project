from utils import parse_tagged_csv
from indexing import Index


def get_tags_tf_idf():
    tags, docs = parse_tagged_csv('phase2_train.csv')
    index = Index("english")
    index.build_with_docs(docs)

    terms = index.index.keys()
    vectors = []
    for doc in docs:
        vector = []
        for term in terms:
            tf = index.get_tf(term, doc['id'], 'text')
            idf = index.get_idf(term, 'text')
            vector.append(tf*idf)
        vectors.append(vector)

    return tags, vectors


tags, vectors = get_tags_tf_idf()
print(tags)

