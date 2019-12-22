from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score

vectors = []
tags = []
with open('vectors.data', 'rb') as f:
    vectors = pickle.load(f)

with open('tags.data', 'rb') as f:
    tags = pickle.load(f)
tags = np.array(tags)
#vectors = np.array(vectors)
n = len(vectors)
n_train = int(0.9 * n)
n_test = n - n_train
train_x = vectors[0:n_train]
train_y = tags[0:n_train]
test_x = vectors[n_train:n]
test_y = tags[n_train:n]


print("#")
knn_doc_mat = train_x



def batch_knn(doc_vectors, k=5):
    input_vectors = doc_vectors
    docs_mat = knn_doc_mat
    score_mat = cosine_similarity(docs_mat, Y=input_vectors)
    score_mat = score_mat.T
    ans = list()
    for i in range(len(doc_vectors)):
        closest = list(reversed(np.argsort(score_mat[i]).tolist()))
        class_count = defaultdict(int)
        for i in range(k):
            current_doc_id = closest[i]
            cls = train_y[current_doc_id]
            class_count[cls] += 1
        max_count = max(list(class_count.values()))
        predicted_classes = [x for x, y in class_count.items() if y == max_count]
        ans.append(predicted_classes[0])
    return ans


def knn(k=9):
    doc_vectors = test_x
    return batch_knn(doc_vectors, k)


pred =knn()
acc = 0
for i in range(len(test_y)):
    if pred[i] == test_y[i]:
        acc += 1

print("Accuracy: ", 1.0 * acc / len(test_y))
print("Precision: ", precision_score(test_y, pred, average=None))
print("Recall: ", recall_score(test_y, pred,  average=None))


# k = 1
#Accuracy:  0.8222222222222222
#Precision:  [0.8173913  0.91363636 0.80578512 0.75      ]
#Recall:  [0.85067873 0.88938053 0.78       0.76847291]

# k = 5
# Accuracy:  0.8588888888888889
# Precision:  [0.84347826 0.93362832 0.84016393 0.815     ]
# Recall:  [0.87782805 0.93362832 0.82       0.80295567]

# k = 9
# Accuracy:  0.86
# Precision:  [0.86936937 0.93362832 0.81568627 0.82233503]
# Recall:  [0.87330317 0.93362832 0.832      0.79802956]