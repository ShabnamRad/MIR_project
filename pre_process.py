import hazm
import itertools
import nltk
from nltk.stem.porter import PorterStemmer
from utils import parse_xml, parse_csv

persian_docs = parse_xml('Persian.xml')
english_docs = parse_csv('English.csv')

persian_stop_words = []
english_stop_words = []


def find_stop_words(docs_data, persian=False):
    word_count = dict()
    doc_fq = dict()
    for doc in docs_data:
        text = "%s %s" % (doc['title'], doc['text'])
        for word in get_tokens(text, persian):
            if is_punctuation(word):
                continue
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
            if word not in doc_fq:
                doc_fq[word] = set()
            doc_fq[word].add(doc['id'])

    doc_count = len(docs_data)
    if persian:
        threshold = sorted(word_count.values())[-15]
    else:
        threshold = sorted(word_count.values())[-30]
    for word in word_count.keys():
        if len(doc_fq[word]) >= 0.50 * doc_count or \
                word_count[word] >= threshold:
            if persian:
                persian_stop_words.append(word)
            else:
                english_stop_words.append(word)


def stem(word, persian=False):
    if persian:
        stemmer = hazm.Stemmer()
    else:
        stemmer = PorterStemmer()
    return stemmer.stem(word)


def get_tokens(raw_text, persian=False):
    if persian:
        normalizer = hazm.Normalizer()
        text = normalizer.normalize(raw_text)
        return list(itertools.chain(*[hazm.word_tokenize(sent) for sent in hazm.sent_tokenize(text)]))
    else:
        tokens = nltk.word_tokenize(raw_text)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w.lower() for w in tokens]
        return tokens


def is_punctuation(token):
    return sum([(c in token) for c in r'\.:!،؛؟»\]\)\}«\[\(\{?*=|-;,']) > 0


def is_stop_word(token, persian=False):
    if persian:
        return token in persian_stop_words
    else:
        return token in english_stop_words


def process(raw_text, persian=False, remove_stop_words=True):
    # normalizer & tokenizer
    tokens = get_tokens(raw_text, persian)

    final_tokens = []
    for token in tokens:
        flag = False

        # remove stop words
        if remove_stop_words:
            flag |= is_stop_word(token, persian)

        # remove punctuations
        flag |= is_punctuation(token)

        if not flag:
            # stemming
            token = stem(token, persian)
            if len(token):
                final_tokens.append(token)


    return final_tokens


find_stop_words(persian_docs, True)
find_stop_words(english_docs)
#print("persian docs stop words: ", persian_stop_words)
#print("english docs stop words: ", english_stop_words)


#print(process("اسم من شبنم است. این یک نام فارسی است!", True))
#print(process("my name is Parand, and it is a Persian name"))
