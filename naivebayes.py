from utils import parse_tagged_csv
from indexing import Index
import pickle
from collections import defaultdict
import math
import pre_process
from sklearn.metrics import precision_score, recall_score

tags, docs = parse_tagged_csv('phase2_train.csv')
print("#")
index = Index("english")
index.build_with_docs(docs)
print("#####")
terms = index.index.keys()

test_tags, test_docs = parse_tagged_csv('phase2_test.csv')
categories = [1, 2, 3, 4]

def trainNB():
    vocab_size = len(index.index.keys())
    tf_class = defaultdict(lambda: defaultdict(int))
    class_tf_sum = defaultdict(int)
    class_docs = defaultdict(int)
    for doc_id, doc_data in index.doc_set.items():
        c = tags[doc_id]
        class_docs[c] += 1
        for word, tf_doc in doc_data['tf'].items():
            tf_class[c][word] += tf_doc
            class_tf_sum[c] += tf_doc

    # fill conditional_probability[t][c]
    conditional_probability = dict()
    for t in index.index.keys():
        conditional_probability[t] = dict()
        for c in categories:
            conditional_probability[t][c] = (tf_class[c][t] + 1) / (vocab_size + class_tf_sum[c])

    # fill class_priority[c]
    class_priority = dict()
    docs_size = len(index.doc_set)
    for c in categories:
        class_priority[c] = class_docs[c] / docs_size
    return class_priority, conditional_probability

print("before train")
class_priority, conditional_probability = trainNB()
print("after train")


def predictNBBatch(docs):
    answers = list()
    for document in docs:
        tokens = pre_process.process(document['text'])

        score = dict()
        for c in categories:
            score[c] = math.log(class_priority[c])
            for t in tokens:
                if t not in index.index.keys():
                    continue
                score[c] += math.log(conditional_probability[t][c])

        predicted_class = max([(y, x) for x, y in score.items()])[1]
        answers.append(predicted_class)
    return answers


pred = predictNBBatch(test_docs)
acc = 0

for i in range(len(test_tags)):
    if pred[i] == test_tags[i]:
        acc += 1

precision = precision_score(test_tags, pred, average=None)
recall = recall_score(test_tags, pred,  average=None)
print("Accuracy: ", 1.0 * acc / len(test_tags))
print("Precision: ", precision)
print("Recall: ", recall)



f1 = []
for i in range(4):
    tmp = (2.0 * precision[i] * recall[i]) / (precision[i] + recall[i])
    f1.append(tmp)

print("F1: ", f1)

#train
# Accuracy:  0.9257777777777778
# Precision:  [0.94922232 0.96336677 0.89238264 0.898365  ]
# Recall:  [0.92222222 0.98177778 0.89555556 0.90355556]


#test:
# Accuracy:  0.858
# Precision:  [0.91101695 0.91119691 0.78461538 0.82857143]
# Recall:  [0.86  0.944 0.816 0.812]
# F1:  [0.8847736625514404, 0.9273084479371315, 0.7999999999999999, 0.8202020202020203]