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
print("Accuracy: ", clf.score(test_x, test_y))
print("Precision: ", precision_score(test_y, pred, average=None))
print("Recall: ", recall_score(test_y, pred,  average=None))


# c = 0.5
#Accuracy:  0.8355555555555556
#Precision:  [0.91346154 0.87398374 0.9273743  0.67790262]
#Recall:  [0.85972851 0.95132743 0.664      0.89162562]

# c= 1.5
#Precision:  [0.92344498 0.93886463 0.9086758  0.74485597]
#Recall:  [0.87330317 0.95132743 0.796      0.89162562]

#c = 2
# Accuracy:  0.8777777777777778
# Precision:  [0.91981132 0.93859649 0.90178571 0.75847458]
# Recall:  [0.88235294 0.94690265 0.808      0.8817734 ]


# for c=1 : Precision:  [0.91549296 0.91561181 0.90990991 0.78070175] Recall:  [0.88235294 0.96017699 0.808 0.87684729]

