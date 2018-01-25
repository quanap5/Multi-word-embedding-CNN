from __future__ import print_function

import os

import numpy as np
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from model.model import model_selector
from model.model_2channel import model_selector2
from model.model_3channel import model_selector3
from model.model_2channel_join import model_selectorBoth
from reader.filereader import read_glove_vectors, read_input_data
from utils import argumentparser
from sklearn import metrics

np.random.seed(42)


def main():
    args = argumentparser.ArgumentParser()
    train(args)


def train(args):
    print('Reading word vectors.')
    #embeddings_index = read_glove_vectors(args.embedding_file_path)

    embeddings_index = read_glove_vectors("/home/duong/Desktop/CNN-Sentence-Classifier/app/GoogleNews-vectors-negative300.txt")
    embeddings_index2 = read_glove_vectors("/home/duong/Desktop/CNN-Sentence-Classifier/app/glove.txt")
    print('Found {} word vectors in embedding2.'.format(len(embeddings_index2)))

    print('Processing input data')
    #texts, labels_index, labels = read_input_data(args.data_dir)
    texts, labels_index, labels, textsPCQ, labels_indexPCQ, labelsPCQ, \
    textsVali, labels_indexVali, labelsVali = read_input_data(args.data_dir)

    # texts - list of text samples
    # labels_index - dictionary mapping label name to numeric id
    # labels - list of label ids
    print('Found {} texts.'.format(len(textsPCQ)))

    # Vectorize the text sample into 2D integer tensor
    tokenizer = Tokenizer(nb_words=args.nb_words)
    tokenizer.fit_on_texts(textsPCQ)
    sequences = tokenizer.texts_to_sequences(textsPCQ)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    data = pad_sequences(sequences, maxlen=args.max_sequence_len)

    # Transform labels to be categorical variables
    labelsPCQ = to_categorical(np.asarray(labelsPCQ))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labelsPCQ.shape)

    #Infor of Validat dataset
    print('Found {} Vali.'.format(len(textsVali)))

    # Vectorize the text sample into 2D integer tensor
    tokenizerVali = Tokenizer(nb_words=args.nb_words)
    tokenizerVali.fit_on_texts(textsVali)
    sequencesVali = tokenizerVali.texts_to_sequences(textsVali)
    word_indexVali = tokenizerVali.word_index
    print('Found {} unique tokens in Vali.'.format(len(word_indexVali)))

    dataVali = pad_sequences(sequencesVali, maxlen=args.max_sequence_len)

    # Transform labels to be categorical variables
    labelsVali = to_categorical(np.asarray(labelsVali))
    print('Shape of data tensor in Vali:', dataVali.shape)
    print('Shape of label tensor in Vali:', labelsVali.shape)

    #split the input data into training set and validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labelsPCQ = labelsPCQ[indices]
    nb_validation_samples = int(args.validation_split * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labelsPCQ[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labelsPCQ[-nb_validation_samples:]

    # indices_train = np.arange(data.shape[0])
    # np.random.shuffle(indices_train)
    # data = data[indices_train]
    # labelsPCQ = labelsPCQ[indices_train]
    #
    # indicesVali = np.arange(dataVali.shape[0])
    # np.random.shuffle(indicesVali)
    # dataVali = dataVali[indicesVali]
    # labelsVali = labelsVali[indicesVali]
    #
    # x_train = data
    # y_train = labelsPCQ
    # x_val = dataVali
    # y_val = labelsVali

    print('Preparing embedding matrix.')

    # initiate embedding matrix with zero vectors for embedding1.
    nb_words = min(args.nb_words, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    args.nb_words = nb_words
    args.len_labels_index = len(labels_index)

    # initiate embedding matrix with zero vectors for embedding2.
    nb_words2 = min(args.nb_words, len(word_index))
    embedding_matrix2 = np.zeros((nb_words2 + 1, args.embedding_dim2))#+100
    for word, i in word_index.items():
        if i > nb_words2:
            continue
        embedding_vector2 = embeddings_index2.get(word)
        if embedding_vector2 is not None:
            embedding_matrix2[i] = embedding_vector2
    args.nb_words = nb_words
    args.len_labels_index = len(labels_index)

    '''Remember uncomment model according to model.fit below'''
    #model = model_selector(args, embedding_matrix)
    #model = model_selector2(args, embedding_matrix, embedding_matrix2)
    model = model_selectorBoth(args, embedding_matrix, embedding_matrix2)

    checkpoint_filepath = os.path.join(args.model_dir, "weights.best.hdf5")
    # checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss',
    #                              verbose=1, save_best_only=True)
    # callbacks_list = [checkpoint]

    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [earlystopper, checkpointer]
    model_json = model.to_json()
    with open(os.path.join(args.model_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)


    #model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=args.num_epochs,
    #          batch_size=args.batch_size, callbacks=callbacks_list)
    model.fit([x_train, x_train], y_train, validation_data=([x_val, x_val], y_val), nb_epoch=args.num_epochs,
              batch_size=args.batch_size, callbacks=callbacks_list)
    #model.fit([x_train, x_train, x_train], y_train, validation_data=([x_val, x_val, x_val], y_val), nb_epoch=args.num_epochs,
    #          batch_size=args.batch_size, callbacks=callbacks_list)
    print("Test model ...")
    print("Loading ...", checkpoint_filepath)
    model.load_weights(checkpoint_filepath)
    y_prob = model.predict([x_val,x_val])
    roc = metrics.roc_auc_score(y_val, y_prob)
    print("ROC Prediction (binary classification):", roc)
    # acc2 = metrics.accuracy_score(y_val, y_prob)
    # print("Raw Accuracy:", acc2)


if __name__ == '__main__':
    main()
