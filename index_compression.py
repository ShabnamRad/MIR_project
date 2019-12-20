import os
import pickle

from indexing import *


def sequence_to_vb_code(seq):
    if not seq:
        return ""

    vb_code = ""
    gap_seq = [seq[0]]
    for i in range(1, len(seq)):
        gap_seq.append(seq[i] - seq[i - 1])

    for num in gap_seq:
        binary_num = bin(num)[2:]

        num_vb_code = ""
        first_bit = "1"
        while len(binary_num) > 7:
            code = first_bit + binary_num[-7:]
            binary_num = binary_num[:-7]
            first_bit = "0"
            num_vb_code = code + num_vb_code

        if len(binary_num) > 0:
            code = first_bit + "0" * (7 - len(binary_num)) + binary_num
            num_vb_code = code + num_vb_code

        vb_code += num_vb_code

    return vb_code


def vb_code_to_sequence(vb_code):
    gap_seq = []
    num_of_bytes = len(vb_code) // 8

    start_byte = 0
    for i in range(num_of_bytes):
        if vb_code[8 * i] == '1':  # last byte of a number
            binary = "0b"
            vb_code_segment = vb_code[8 * start_byte: 8 * (i + 1)]  # a vb_code segment representing one number
            num_of_bytes_of_number = len(vb_code_segment) // 8

            for j in range(num_of_bytes_of_number):
                binary += vb_code_segment[(8 * j + 1): 8 * (j + 1)]

            gap_seq.append(int(binary, 2))
            start_byte = i + 1

    seq = [gap_seq[0]]
    for i in range(1, len(gap_seq)):
        seq.append(seq[i - 1] + gap_seq[i])

    return seq


def sequence_to_gamma_code(seq):
    if not seq:
        return ""

    gamma_code = ""
    gap_seq = [seq[0]]
    for i in range(1, len(seq)):
        gap_seq.append(seq[i] - seq[i - 1])

    gap_seq[0] += 1  # preventing 0
    for num in gap_seq:
        offset = bin(num)[3:]  # excluding 0b1
        length = "1" * len(offset) + "0"

        gamma_code += length + offset

    return gamma_code


def gamma_code_to_sequence(gamma_code):
    gap_seq = []

    traverser, length = 0, 0
    while len(gamma_code) > 0:
        if gamma_code[traverser] == "1":
            traverser += 1
            length += 1
        else:
            gamma_code = gamma_code[traverser:]
            binary_number = "0b1" + gamma_code[:length + 1]
            gap_seq.append(int(binary_number, 2))

            gamma_code = gamma_code[length + 1:]
            traverser, length = 0, 0

    gap_seq[0] -= 1
    seq = [gap_seq[0]]
    for i in range(1, len(gap_seq)):
        seq.append(seq[i - 1] + gap_seq[i])

    return seq


def compress(posting_lists, method):
    res = posting_lists.copy()
    for (term, docs_list) in posting_lists.items():
        compressed_docs_list = docs_list.copy()
        for (docId, positions_list) in docs_list.items():
            compressed_positions_list = positions_list.copy()
            if 'text' in positions_list.keys():
                if method == "gamma":
                    compressed_list = sequence_to_gamma_code(positions_list['text'])
                else:
                    compressed_list = sequence_to_vb_code(positions_list['text'])
                compressed_positions_list.update({'text': compressed_list})
            if 'title' in positions_list.keys():
                if method == "gamma":
                    compressed_list = sequence_to_gamma_code(positions_list['title'])
                else:
                    compressed_list = sequence_to_vb_code(positions_list['title'])
                compressed_positions_list.update({'title': compressed_list})
            compressed_docs_list.update({docId: compressed_positions_list})
            res.update({term: compressed_docs_list})
    return res


def compare_file_sizes(language):
    if language == "english":
        index = Index("english")
        index = index.load_from_file("index-en.pkl")
        indices = index.index
        print("English indices:")
        print(" size of posting_lists before compression =", os.path.getsize("index-en.pkl"), "bytes")
        vb_compressed_pl = compress(indices, "vb")
        gamma_compressed_pl = compress(indices, "gamma")
        with open("index-en-vb-compressed.pkl", "wb") as f:
            pickle.dump(vb_compressed_pl, f)
        with open("index-en-gamma-compressed.pkl", "wb") as f:
            pickle.dump(gamma_compressed_pl, f)
        print(" size of posting_lists after variable byte-code compression =",
              os.path.getsize("index-en-vb-compressed.pkl"), "bytes")
        print(" size of posting_lists after gamma code compression =", os.path.getsize("index-en-gamma-compressed.pkl"),
              "bytes")
    else:
        index = Index("persian")
        index = index.load_from_file("index-fa.pkl")
        indices = index.index
        print("Persian indices:")
        print(" size of posting_lists before compression =", os.path.getsize("index-fa.pkl"), "bytes")
        vb_compressed_pl = compress(indices, "vb")
        gamma_compressed_pl = compress(indices, "gamma")
        with open("index-fa-vb-compressed.pkl", "wb") as f:
            pickle.dump(vb_compressed_pl, f)
        with open("index-fa-gamma-compressed.pkl", "wb") as f:
            pickle.dump(gamma_compressed_pl, f)
        print(" size of posting_lists after variable byte-code compression =",
              os.path.getsize("index-fa-vb-compressed.pkl"), "bytes")
        print(" size of posting_lists after gamma code compression =", os.path.getsize("index-fa-gamma-compressed.pkl"),
              "bytes")


def load_and_print_compressed_indices(language):
    if language == "persian":
        with open("index-fa-vb-compressed.pkl", "rb") as f:
            data = pickle.load(f)
        print(data)
    else:
        with open("index-en-vb-compressed.pkl", "rb") as f:
            data = pickle.load(f)
        print(data)
