import scipy.cluster.hierarchy as shc
import pickle
import numpy as np

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
vectors = []
tags = []
with open('word2vec.data', 'rb') as f:
    vectors = pickle.load(f)


with open('ids.data', 'rb') as f:
    ids = pickle.load(f)

vectors = np.array(vectors)
ids = np.array(ids)
n, m = vectors.shape

def test_param():
    plt.figure(figsize=(10, 7))
    plt.title("TfIDF")
    dend = shc.dendrogram(shc.linkage(vectors, method='ward'))
    plt.show()

def test(k):
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    pred = cluster.fit_predict(vectors)

    print(pred)

    with open('h-w2v.csv', mode='w') as f:
        for i in range(len(ids)):
            f.write(str(ids[i]) + ", " + str(pred[i]) + "\n")


test(11)