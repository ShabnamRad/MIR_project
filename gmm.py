import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

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
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='diag', verbose=True).fit(vectors)
              for n in n_components]

    plt.plot(n_components, [m.bic(vectors) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(vectors) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.show()

def test(k):
    gmm = GaussianMixture(k, covariance_type='diag', verbose=True).fit(vectors)
    pred = gmm.predict(vectors)
    print(pred)

    with open('gmm-w2v.csv', mode='w') as f:
        for i in range(len(ids)):
            f.write(str(ids[i]) + ", " + str(pred[i]) + "\n")

test(15)


#tfidf = 11
#w2v 15