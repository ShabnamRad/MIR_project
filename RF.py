import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import numpy as np


vectors = []
tags = []

with open('vectors.data', 'rb') as f:
    vectors = pickle.load(f)


with open('tags.data', 'rb') as f:
    tags = pickle.load(f)

tags = np.array(tags)


test_data = []
test_tags = []

with open('test_vectors.data', 'rb') as f:
    test_data = pickle.load(f)
with open('test_tags.data', 'rb') as f:
    test_tags = pickle.load(f)

def random_forrest(vectors_to_predict=vectors, true_tags = tags):
    print("#")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(vectors, tags)
    pred = clf.predict(vectors_to_predict)
    print("Accuracy: ", clf.score(vectors_to_predict, true_tags))
    return pred


pred = random_forrest(test_data, test_tags)

precision =  precision_score(test_tags, pred, average=None)
recall = recall_score(test_tags, pred,  average=None)
print("Precision: ", precision)
print("Recall: ", recall)


f1 = []
for i in range(4):
    tmp = (2.0 * precision[i] * recall[i]) / (precision[i] + recall[i])
    f1.append(tmp)

print("F1: ", f1)
# train:
# Accuracy:  0.9933333333333333
# Precision:  [0.99333629 0.99204596 0.9915518  0.99284756]
# Recall:  [0.99377778 0.99777778 0.99111111 0.98711111]


# test:
# Accuracy:  0.749
# Precision:  [0.76422764 0.81617647 0.692      0.71551724]
# Recall:  [0.752 0.888 0.692 0.664]
# F1:  [0.7580645161290323, 0.8505747126436781, 0.692, 0.6887966804979253]