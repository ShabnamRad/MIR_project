import pre_process
import indexing
from indexing import *
import index_compression
import utils

index = None
persian_docs = utils.parse_xml('Persian.xml')
english_docs = utils.parse_csv('English.csv')


def process_input(q, lang):
    if lang == 'persian':
        return pre_process.process(q, True)
    return pre_process.process(q)


def print_stopwords():
    pre_process.find_stop_words(persian_docs, True)
    pre_process.find_stop_words(english_docs)
    print("persian docs stop words: ", pre_process.persian_stop_words)
    print("english docs stop words: ", pre_process.english_stop_words)


def print_posting_list(word):
    print(index.get_posting_list(word))


def print_posting_list_position(word, doc):
    print(index.get_posting_list_position(word, doc))


def print_compression(yes, lang):
    if lang == "persian":
        index_compression.compare_file_sizes("persian")
        if yes:
            index_compression.load_and_print_compressed_indices()
    else:
        index_compression.compare_file_sizes("english")
        if yes:
            index_compression.load_and_print_compressed_indices()


def spell_correction(q):
    print(indexing.spell_correction(q, index))


if __name__ == '__main__':

    while True:
        lang = input("language:")
        if lang == "persian":
            index = indexing.Index("persian")
            break
        elif lang == "english":
            index = indexing.Index("english")
            break
        else:
            print("Unknown Language")
    index.build()

    while True:
        fun = input("function:")
        if fun == 'done':
            break
        if fun == 'process':
            q = input("query:")
            print(process_input(q, lang))
        elif fun == 'stop_words':
            print_stopwords()
        elif fun == 'post':
            q = input("query:")
            print_posting_list(q)
        elif fun == 'post_pos':
            q = input("query:")
            doc = int(input("doc:"))
            print_posting_list_position(q, doc)
        elif fun == 'compress':
            q = input("print compressed indices?(y/n)")
            print_compression(q == "y", lang)
        elif fun == 'correct':
            q = input("query:")
            print(spell_correction(q))
        elif fun == 'search':
            q = input("query:")
            t = 0
            if lang == "english":
                t = input("subject:")
            print(indexing.search(q, t, index))
