from __future__ import print_function

import os

import numpy as np
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from model.model import model_selector
from model.model_2channel import model_selector2
from model.model_3channel import model_selector3
from reader.filereader_ori import read_glove_vectors, read_input_data
from utils import argumentparser
from sklearn import metrics

#np.random.seed(42)
seed=42

def main():
    args = argumentparser.ArgumentParser()
    train(args)


def train(args):
    print('Reading word vectors.')
    #embeddings_index = read_glove_vectors(args.embedding_file_path)

    embeddings_index = read_glove_vectors("/home/duong/Desktop/CNN-Sentence-Classifier/app/ORIcrisis_embeddings.txt")
    #embeddings_index2 = read_glove_vectors("/home/duong/Desktop/CNN-Sentence-Classifier/app/glove2.txt")
    print('Found {} word vectors in emdedding1.'.format(len(embeddings_index)))
    #print('Found {} word vectors in embedding2.'.format(len(embeddings_index2)))

    print('Processing input data')

    input_name = ["input_CR_prccd.txt", "input_Sub_prccd.txt", "input_MPQA_prccd.txt", "inputPCQM_prccd.txt",
                  "input_flood_phi_prccd.txt", "input_flood_colorado_prccd.txt", "input_flood_qeen_prccd.txt",
                  "input_flood_manila_prccd.txt", "input_fire_australia_prccd.txt", "input_earthquake_chile_prccd.txt"]
    label_name = ["label_CR.txt", "label_input_Sub.txt", "label_MPQA.txt", "labelPCQM.txt", "label_flood_phi.txt",
                  "label_flood_colorado.txt", "label_flood_qeen.txt", "label_flood_manila.txt",
                  "label_fire_australia.txt", "label_earthquake_chile.txt"]

    with open("11Janlan01_crosstrainBest_CV50_CrisEmb_cnn1xStatic.txt", 'wb') as result_CV:
        for list in range(0, 10):
            texts, labels_index, labels = read_input_data(args.data_dir, input_name[list],label_name[list])
            # texts - list of text samples
            # labels_index - dictionary mapping label name to numeric id
            # labels - list of label ids
            print('Found {} texts.'.format(len(texts)))

            # Vectorize the text sample into 2D integer tensor
            tokenizer = Tokenizer(num_words=args.nb_words)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            word_index = tokenizer.word_index
            print('Found {} unique tokens.'.format(len(word_index)))

            data = pad_sequences(sequences, maxlen=args.max_sequence_len)

            # Transform labels to be categorical variables
            labels = to_categorical(np.asarray(labels))
            print('Shape of data tensor:', data.shape)
            print('Shape of label tensor:', labels.shape)

            # split the input data into training set and validation set
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]

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


            #   return architecture of model
            model = model_selector(args, embedding_matrix)

            #   Loop through the indices the split () method return

            cv_scores = []
            ROC_scores = []
            fold=10

            for i in range(0,fold):
                print("\n")
                print("\n")
                print("\n")
                print("-------------FOLD :",(i+1))
                window_data=data.shape[0]/fold
                #   Generate batches from indices
                x_train1 = data[:i*window_data]
                x_train2 = data[(i+1)*window_data:]

                y_train1 = labels[:i * window_data]
                y_train2 = labels[(i + 1) * window_data:]

                if i==0:
                    x_trainAll = x_train2
                    y_trainAll = y_train2
                else:
                    x_trainAll = np.concatenate((x_train1,x_train2),axis=0)
                    y_trainAll = np.concatenate((y_train1,y_train2), axis=0)



                x_val = data[i*window_data:(i+1)*window_data]
                y_val = labels[i*window_data:(i+1)*window_data]

                indices_ = np.arange(x_trainAll.shape[0])
                np.random.shuffle(indices_)
                x_train= x_trainAll[indices_]
                y_train = y_trainAll[indices_]
                nb_validation_samples = int(args.validation_split * x_train.shape[0])

                x_train = x_train[:-nb_validation_samples]
                y_train = y_train[:-nb_validation_samples]
                x_dev = x_train[-nb_validation_samples:]
                y_dev = y_train[-nb_validation_samples:]



                #   Clear model and create
                model=None
                model = model_selector(args, embedding_matrix)

                checkpoint_filepath = os.path.join(args.model_dir, "weights.best.hdf5")
                earlystopper = EarlyStopping(monitor='val_acc', patience=3, verbose=0)
                checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=0, save_best_only=True)
                callbacks_list = [earlystopper, checkpointer]
                model_json = model.to_json()
                with open(os.path.join(args.model_dir, "model.json"), "w") as json_file:
                    json_file.write(model_json)

                #model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)
                model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=args.num_epochs,
                          batch_size=args.batch_size, callbacks=callbacks_list)
                y_prob = model.predict(x_val)



                roc = metrics.roc_auc_score(y_val, y_prob)
                print("ROC Prediction (binary classification):", roc)
                scores = model.evaluate(x_val, y_val, verbose=0)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                cv_scores.append(scores[1] * 100)
                ROC_scores.append(roc * 100)

            print(input_name[list])
            print("ACC: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
            print("ROC: %.2f%% (+/- %.2f%%)" % (np.mean(ROC_scores), np.std(ROC_scores)))
            result_CV.write(input_name[list] + " ACC: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)) + " ROC: %.2f%% (+/- %.2f%%)" % (np.mean(ROC_scores), np.std(ROC_scores)) + '\n')
            result_CV.write(time.asctime(time.localtime(time.time())) + '\n')

if __name__ == '__main__':
    main()
