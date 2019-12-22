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

def random_forrest(vectors_to_predict=vectors):
    print("#")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(vectors, tags)
    pred = clf.predict(vectors_to_predict)
    return pred


pred = random_forrest()
print("Precision: ", precision_score(tags, pred, average=None))
print("Recall: ", recall_score(tags, pred, average=None))

# Precision:  [0.99333629 0.99204596 0.9915518  0.99284756]
# Recall:  [0.99377778 0.99777778 0.99111111 0.98711111]
