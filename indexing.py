import pickle
import math
import pre_process
import utils


class BigramIndex(object):
    def __init__(self):
        self.words = set()
        self.index = dict()

    def add_word(self, word):
        if word in self.words:
            return

        for i in range(len(word) - 1):
            part = word[i:i + 2]
            if part not in self.index:
                self.index[part] = {word}
            if word not in self.index[part]:
                self.index[part].add(word)

        self.words.add(word)

    def remove_word(self, word):
        if word not in self.words:
            return

        for i in range(len(word) - 1):
            part = word[i:i + 2]
            self.index[part].remove(word)
            if len(self.index[part]) is 0:
                self.index.pop(part)

        self.words.remove(word)

    def get_bigram(self, bigram):
        return self.index.get(bigram, None)


class Index(object):
    def __init__(self, lang):
        self.index = dict()
        self.bigram = BigramIndex()
        self.doc_set = dict()
        self.df = dict()
        self.ispersian = False
        if lang == 'persian':
            self.ispersian = True
        self.raw_docs = []
        self.doc_similarity = dict()


    def get_tf(self, token, doc_id, label = 'text'):
        if token not in self.index:
            return 0
        if doc_id not in self.index[token]:
            return 0
        return len(self.index[token][doc_id].get(label, []))

    def get_idf(self, token, label):
        num = 0
        if label is None:
            num = len(self.index.get(token))
        elif token in self.index:
            num = self.df[token].get(label, 0)
        if num == 0:
            return 0
        idf = math.log(1.0 * len(self.doc_set) / num)
        return idf

    def get_posting_list(self, word):
        return self.index.get(word, None)

    def get_posting_list_position(self, word, doc):
        posting_list = self.get_posting_list(word)
        for item in posting_list.items():
            if item[0] == doc:
                return item[1]
        return None

    def add(self, doc_id, label, text):
        tokens = pre_process.process(text, self.ispersian, remove_stop_words=False)

        count = dict()
        for i, token in enumerate(tokens):
            if token not in self.index:
                self.bigram.add_word(token)
                self.index[token] = dict()
            if doc_id not in self.index[token]:
                self.index[token][doc_id] = dict()
            if label not in self.index[token][doc_id]:
                self.index[token][doc_id][label] = list()
                if token not in self.df:
                    self.df[token] = dict()
                if label not in self.df[token]:
                    self.df[token][label] = 0
                self.df[token][label] += 1
            self.index[token][doc_id][label].append(i)

            if token not in count:
                count[token] = 0
            count[token] = count[token] + 1

        if doc_id not in self.doc_similarity:
            self.doc_similarity[doc_id] = dict()
        self.doc_similarity[doc_id][label] = math.sqrt(sum([v * v for v in count.values()]))
        return tokens

    def add_doc(self, doc):
        doc_id = doc['id']
        if doc_id in self.doc_set.keys():
            return

        self.doc_set[doc_id] = dict()
        self.add(doc_id, 'title', doc['title'])
        tokens = self.add(doc_id, 'text', doc['text'])

        self.doc_set[doc_id]['tf'] = dict()
        for token in set(tokens):
            self.doc_set[doc_id]['tf'][token] = self.get_tf(token, doc_id)

    def remove_doc(self, doc):
        doc_id = doc['id']
        if not doc_id in self.doc_set.keys():
            return
        self.doc_set.pop(doc_id)
        self.doc_similarity.pop(doc_id)
        tokens = pre_process.process("%s %s" % (doc['title'], doc['text']), self.ispersian, remove_stop_words=False)
        for token in set(tokens):
            for k, v in self.index[token][doc_id].items():
                self.df[token][k] -= 1
            self.index[token].pop(doc_id)
            if len(self.index[token]) == 0:
                self.index.pop(token)
                self.bigram.remove_word(token)

    def build(self):
        if self.ispersian:
            self.raw_docs = utils.parse_xml("Persian.xml")
        else:
            self.raw_docs = utils.parse_csv("English.csv")
        for doc in self.raw_docs:
            self.add_doc(doc)

    def build_with_docs(self, docs):
        self.raw_docs = docs
        for doc in self.raw_docs:
            self.add_doc(doc)

    def save_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def spell_correction(query, index):
    tokens = pre_process.process(query)
    ans = [correction(token, index) for token in tokens]
    return ' '.join(ans)


def correction(token, index):
    if token in index.index:
        return token
    similars = jaccard(token, index)

    min_dis = None
    ans = None
    for t in similars:
        dist = edit_distance(token, t)
        if ans is None or dist < min_dis:
            min_dis = dist
            ans = t

    return ans


def jaccard(token, index):
    bigrams = set([token[i:i + 2] for i in range(len(token) - 1)])
    score = dict()
    for bi in bigrams:
        words = index.bigram.get_bigram(bi)
        if words is None:
            continue
        for w in words:
            if w not in score:
                x = set([token[i:i + 2] for i in range(len(token) - 1)])
                y = set([w[i:i + 2] for i in range(len(w) - 1)])
                score[w] = 1.0 * len(x.intersection(y)) / (len(x) + len(y) - len(x.intersection(y)))

    if len(score) == 0:
        return list(index.index.keys())
    best = max(score.values())
    return [k for k, v in score.items() if v == best]


def edit_distance(x, y):
    n, m = len(x), len(y)
    d = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
            if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def search(query, tag, index):
    if tag:
        with open('phase1_tags.data', 'rb') as f:
            tags = pickle.load(f)
    tokens = spell_correction(query, index).split()
    docs = set()
    for w in tokens:
        posting_list = index.get_posting_list(w)
        posting_list = set(posting_list.keys())
        if tag:
            posting_list = set(filter(lambda x: tags[x] == tag, posting_list))
        docs = docs.union(posting_list)
    scores = {}

    q = dict()
    for token in tokens:
        if token not in q:
            q[token] = 0
        q[token] += 1
    q_cosine = math.sqrt(sum([v * v for v in q.values()]))

    for doc in docs:
        s = 0
        for token in tokens:
            tf = index.get_tf(token, doc, 'text')
            if tf == 0:
                continue
            tf_idf = (1 + math.log(tf)) * index.get_idf(token, 'text') * (
                        1 + math.log(sum([token == t for t in tokens])))
            tf_idf /= index.doc_similarity[doc]['text'] * q_cosine
            s += tf_idf

        scores[doc] = s
    ans = list(sorted([(v, k) for k, v in scores.items()], reverse=True))[:10]
    return [(v, k) for k, v in ans if k > 0.1]
