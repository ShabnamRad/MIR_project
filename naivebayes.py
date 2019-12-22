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

def predictNB(document):
    return predictNBBatch([document])[0]


pred = []
acc = 0
i = 0
for doc in docs:
    if i % 100 == 0:
        print(i)
    i += 1
    p = predictNB(doc)
    pred.append(p)


for i in range(len(tags)):
    if pred[i] == tags[i]:
        acc += 1

print("Accuracy: ", 1.0 * acc / len(tags))
print("Precision: ", precision_score(tags, pred, average=None))
print("Recall: ", recall_score(tags, pred,  average=None))

# Accuracy:  0.9257777777777778
# Precision:  [0.94922232 0.96336677 0.89238264 0.898365  ]
# Recall:  [0.92222222 0.98177778 0.89555556 0.90355556]