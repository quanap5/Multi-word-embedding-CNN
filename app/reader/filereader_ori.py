from __future__ import division
from __future__ import print_function

import os

import numpy as np


def read_glove_vectors(glove_vector_path):
    '''Method to read glove vectors and return an embedding dict.'''
    embeddings_index = {}  # dictionary mapping word to numberical vector
    with open(glove_vector_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs[:]
    return embeddings_index


def read_input_data(input_data_path, input_='', label_=''):
    """Method to read data from input_data_path"""
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    texts = list(open(os.path.join(input_data_path, input_), "r").readlines())

    with open(os.path.join(input_data_path, label_), 'r') as label_f:
        largest_label_id = 0
        for line in label_f:
            label = str(line.strip())
            if label not in labels_index:  # check label and add if not vailable
                labels_index[label] = largest_label_id
                largest_label_id += 1
            labels.append(labels_index[label])

    '''Method to read data PCQ as training dataset'''
    #
    # labels_indexPCQ = {}  # dictionary mapping label name to numeric id
    # labelsPCQ = []  # list of label ids
    # textsPCQ = list(open(os.path.join(input_data_path, "inputPCQM.txt"), "r").readlines())
    #
    # with open(os.path.join(input_data_path, "labelPCQM.txt"), 'r') as label_fPCQ:
    #     largest_label_idPCQ = 0
    #     for line in label_fPCQ:
    #         labelPCQ = str(line.strip())
    #         if labelPCQ not in labels_indexPCQ:  # check label and add if not vailable
    #             labels_indexPCQ[labelPCQ] = largest_label_idPCQ
    #             largest_label_idPCQ += 1
    #         labelsPCQ.append(labels_indexPCQ[labelPCQ])
    #
    # '''Method to read validate data '''
    #
    # labels_indexVali = {}  # dictionary mapping label name to numeric id
    # labelsVali = []  # list of label ids
    # textsVali = list(open(os.path.join(input_data_path, "input_flood_colorado_prccd.txt"), "r").readlines())
    #
    # with open(os.path.join(input_data_path, "label_flood_colorado.txt"), 'r') as label_fVali:
    #     largest_label_idVali = 0
    #     for line in label_fVali:
    #         labelVali = str(line.strip())
    #         if labelVali not in labels_indexVali:  # check label and add if not vailable
    #             labels_indexVali[labelVali] = largest_label_idVali
    #             largest_label_idVali += 1
    #         labelsVali.append(labels_indexVali[labelVali])

    return texts, labels_index, labels#, textsPCQ, labels_indexPCQ, labelsPCQ, textsVali, labels_indexVali, labelsVali
