from __future__ import print_function

from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten, Dropout, Merge, Activation, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, RMSprop
from urllib3.poolmanager import pool_classes_by_scheme


def model_selectorBoth(args, embedding_matrix, embedding_matrix2):
    '''Method to select the model to be used for classification'''
    if (args.model_name.lower() != 'self'):
        return _predefined_model(args, embedding_matrix, embedding_matrix2)


def _predefined_model(args, embedding_matrix, embedding_matrix2):
    '''function to use one of the predefined models (based on the paper)'''
    (filtersize_list, number_of_filters_per_filtersize, pool_length_list,
     dropout_list, optimizer, use_embeddings, embeddings_trainable) \
        = _param_selector(args)

    #################branch1########################################
    print('Defining Branch1.')

    input_node = Input(shape=(args.max_sequence_len, args.embedding_dim))
    conv_list = []
    for index, filtersize in enumerate(filtersize_list):
        nb_filter = number_of_filters_per_filtersize[index]
        pool_length = pool_length_list[index]
        #conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(input_node)
        conv = Conv1D(activation='relu', kernel_size = filtersize, filters=nb_filter)(input_node)
        pool = MaxPooling1D(pool_size=pool_length)(conv)
        conv_ = Conv1D(activation='relu', kernel_size=filtersize, filters=nb_filter)(pool)
        pool_ = MaxPooling1D(pool_size=pool_length)(conv_)
        conv__ = Conv1D(activation='relu', kernel_size=filtersize, filters=nb_filter)(pool_)
        pool__ = MaxPooling1D(pool_size=pool_length)(conv__)
        flatten = Flatten()(pool__)
        conv_list.append(flatten)

    if (len(filtersize_list) > 1):
        out = Merge(mode='concat')(conv_list)

    else:
        out = conv_list[0]

    ####################add one more channels as word emmbedding#####################


    if (use_embeddings):
        embedding_layer_1 = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_sequence_len,
                                    trainable=embeddings_trainable)
    else:
        embedding_layer_1 = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_sequence_len,
                                    trainable=embeddings_trainable)


    graph = Model(input=input_node, output=out)

    model = Sequential()

    #model.add(embedding_layer)
    model.add(embedding_layer_1)
    #model.add(Reshape((args.embedding_dim,)))  # (None, 1, 128) -> (None, 128)

    model.add(Dropout(dropout_list[0], input_shape=(args.max_sequence_len, args.embedding_dim)))
    model.add(graph)

    ####################branch2##########################################

    print('Defining Branch2.')

    input_node2 = Input(shape=(args.max_sequence_len, args.embedding_dim2))
    conv_list2 = []
    for index2, filtersize2 in enumerate(filtersize_list):
        nb_filter2 = number_of_filters_per_filtersize[index2]
        pool_length2 = pool_length_list[index2]
        # conv = Conv1D(nb_filter=nb_filter, filter_length=filtersize, activation='relu')(input_node)
        conv2 = Conv1D(activation='relu', kernel_size=filtersize2, filters=nb_filter2)(input_node2)
        pool2 = MaxPooling1D(pool_size=pool_length2)(conv2)
        conv_2 = Conv1D(activation='relu', kernel_size=filtersize, filters=nb_filter)(pool2)
        pool_2 = MaxPooling1D(pool_size=pool_length)(conv_2)
        conv__2 = Conv1D(activation='relu', kernel_size=filtersize, filters=nb_filter)(pool_2)
        pool__2 = MaxPooling1D(pool_size=pool_length)(conv__2)
        flatten2 = Flatten()(pool__2)
        conv_list2.append(flatten2)

    if (len(filtersize_list) > 1):
        out2 = Merge(mode='concat')(conv_list2)
    else:
        out2 = conv_list2[0]

    ####################add one more channels as word emmbedding#####################


    if (use_embeddings):
        embedding_layer_2 = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_sequence_len,
                                    trainable=embeddings_trainable)
    else:
        embedding_layer_2 = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_sequence_len,
                                    trainable=embeddings_trainable)


    graph2 = Model(input=input_node2, output=out2)

    model2 = Sequential()

    # model.add(embedding_layer)
    model2.add(embedding_layer_2)
    # model.add(Reshape((args.embedding_dim,)))  # (None, 1, 128) -> (None, 128)

    model2.add(Dropout(dropout_list[0], input_shape=(args.max_sequence_len, args.embedding_dim2)))
    model2.add(graph2)
    ##############################################
    modelBoth = Sequential()
    modelBoth.add(Merge([model, model2], mode='concat'))

    modelBoth.add(Dense(150))
    modelBoth.add(Dropout(dropout_list[1]))
    modelBoth.add(Activation('relu'))
    modelBoth.add(Dense(args.len_labels_index, activation='softmax'))
    modelBoth.summary()
    modelBoth.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return modelBoth


def _param_selector(args):
    '''Method to select parameters for models defined in Convolutional Neural Networks for
        Sentence Classification paper by Yoon Kim'''
    filtersize_list = [3, 4, 5]
    number_of_filters_per_filtersize = [100, 100, 100]
    pool_length_list = [2, 2, 2]
    dropout_list = [0.5, 0.5]
    optimizer = Adadelta(clipvalue=3)
    use_embeddings = True
    embeddings_trainable = False

    if (args.model_name.lower() == 'cnn-rand'):
        use_embeddings = False
        embeddings_trainable = True
    elif (args.model_name.lower() == 'cnn-static'):
        pass
    elif (args.model_name.lower() == 'cnn-non-static'):
        embeddings_trainable = True
    else:
        filtersize_list = [3, 4, 5]
        number_of_filters_per_filtersize = [150, 150, 150]
        pool_length_list = [2, 2, 2]
        dropout_list = [0.25, 0.5]
        optimizer = RMSprop(lr=args.learning_rate, decay=args.decay_rate,
                            clipvalue=args.grad_clip)
        use_embeddings = True
        embeddings_trainable = True
    return (filtersize_list, number_of_filters_per_filtersize, pool_length_list,
            dropout_list, optimizer, use_embeddings, embeddings_trainable)
