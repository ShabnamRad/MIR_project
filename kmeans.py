import pickle
import numpy as np
from sklearn.cluster import KMeans
import csv
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

def find_param():
    sse = []
    list_k = []
    for k in range(2, 40):

        km = KMeans(n_clusters=k)
        km.fit(vectors)
        sse.append(km.inertia_)
        list_k.append(k)
        print(k, km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.show()

def test(k):

    km = KMeans(n_clusters=k)
    km.fit(vectors)
    pred = km.predict(vectors)
    print(pred)

    with open('kmean-word2vec.csv', mode='w') as f:
        for i in range(len(ids)):
            f.write(str(ids[i]) + ", " + str(pred[i]) + "\n")
test(10)