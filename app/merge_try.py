import numpy as np
from keras.layers import Dense, Input, Convolution1D, MaxPooling1D, Flatten, Activation
from keras.layers import Merge, Embedding
from keras import backend as K
from keras.models import Model, Sequential
vobabsize = 1000
embedding_size = 256
max_pos = 1000
data_len = 200
dim_pose = 100
num_filters = 16
filter_size = 5
poolsize = 2

U = np.random.random((vobabsize, embedding_size))
U_Pose = np.random.random((max_pos+1, dim_pose))

text_branch = Sequential()
text_branch.add(Embedding(vobabsize, embedding_size, input_length=data_len,weights=[U]))

e1pos_branch = Sequential()
e1pos_branch.add(Embedding(max_pos+1, dim_pose, input_length=data_len,weights=[U_Pose]))

e2pos_branch = Sequential()
e2pos_branch.add(Embedding(max_pos+1, dim_pose, input_length=data_len,weights=[U_Pose]))

merged = Merge([text_branch, e1pos_branch, e2pos_branch], mode='concat')
#merged = Merge([merged_m, e2pos_branch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Convolution1D(nb_filter=num_filters,
                         filter_length=filter_size,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
final_model.add(MaxPooling1D(pool_length=poolsize, stride=None, border_mode='valid'))
flatten = Flatten(name="flatten")
final_model.add(flatten)
final_model.add(Dense(3))

final_model.add(Activation('softmax'))
final_model.summary()
final_model.save_weights('finalmodel.h5', overwrite=True)

final_model.load_weights('finalmodel.h5')
print("success")