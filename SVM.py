import pickle
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
import numpy as np
vectors = []
tags = []
with open('vectors.data', 'rb') as f:
    vectors = pickle.load(f)


with open('tags.data', 'rb') as f:
    tags = pickle.load(f)

tags = np.array(tags)
vectors = np.array(vectors)
n, m = vectors.shape
n_train = int(0.9 * n)
n_test = n - n_train
train_x = vectors[0:n_train]
train_y = tags[0:n_train]
test_x = vectors[n_train:n]
test_y = tags[n_train:n]

print("#")
clf = SVC(C=1)
clf.fit(train_x, train_y)
print("#")
pred = clf.predict(test_x)
print("Precision: ", precision_score(test_y, pred, average=None))
print("Recall: ", recall_score(test_y, pred,  average=None))


# for c=1 : Precision:  [0.91549296 0.91561181 0.90990991 0.78070175] Recall:  [0.88235294 0.96017699 0.808 0.87684729]
