import os
longTable=os.environ['longTable']
driam=os.environ['driam']
path=os.environ['path']


#Import libraries
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
from collections import Counter
import random

#Seeding values
from numpy.random import seed
seed(1)
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
tf.config.run_functions_eagerly(True)

#Reading the data into a dataframe
hd = pd.read_csv(path+'spect_drugEmbFingerPrint_Resist_train.csv')
hd=hd.drop(['sample_id'], axis = 1) # Remove column name 'sample_id'
hd=hd.drop(['drug'], axis = 1) # Remove column name 'drug'
hd=hd.drop(['species'], axis = 1) # Remove column name 'species'
hd=hd.drop(['dataset'], axis = 1) # Remove column name 'dataset' (i.e. DRIAMS)
hd.head(10)

#Data exploration
print('Total observations: ', hd.shape[0])
print('Total attributes: ', hd.shape[1] - 1)
chdc = Counter(hd['response'])
print('Total resistant : ', chdc[1]) 
print('Total sensitive : ', chdc[0]) 
resistant=chdc[1]-1 #Number of resistant combinations - 1
sensitive=chdc[0]-1 #Number of sensitive combinations - 1

#Generating pairs for the Siamese network
hd1 = hd[hd['response'] == 1.0].astype('float32')
hd0 = hd[hd['response'] == 0.0].astype('float32')

hd1x = hd1.iloc[:, 1:]
hd1y = hd1.iloc[:, 0]
hd0x = hd0.iloc[:, 1:]
hd0y = hd0.iloc[:, 0]

hd1x = hd1x.to_numpy()
hd1y = hd1y.to_numpy()
hd0x = hd0x.to_numpy()
hd0y = hd0y.to_numpy()

p0 = []
p1 = []
label = []

for i in range(50000):
    t1 = random.randint(0, resistant) #Number of resistant combinations - 1
    t2 = random.randint(0, sensitive) #Number of sensitive combinations - 1
    p0 += [[hd0x[t2], hd1x[t1]]]

for i in range(25000):
    #create pairs sensitive-sensitive
    t1 = random.randint(0, round(sensitive/2)) #(Number of sensitive combinations - 1)/2
    t2 = random.randint(round(sensitive/2), sensitive) #Number of sensitive combinations - 1
    p1 += [[hd0x[t1], hd0x[t2]]]
    #create pairs resistant-resistant
    t1 = random.randint(0, round(resistant/2)) #(Number of resistant combinations - 1)/2
    t2 = random.randint(round(resistant/2), resistant)  #Number of resistant combinations - 1
    p1 += [[hd1x[t1], hd1x[t2]]]

#Gather all created pairs
p = []
for i in range(50000):
    p.append(p0[i])
    label.append(0)
    p.append(p1[i])
    label.append(1)

X = np.array(p)
Y = np.array(label)

for _ in range(100):
    p = np.random.permutation(100000)
    X = X[p]
    Y = Y[p]

x_train = X[:80000]
x_test = X[80000:]
y_train = Y[:80000]
y_test = Y[80000:]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train[10:20])

cc = Counter(y_test)
print(cc[1])
print(cc[0])

#Creating and compiling the neural network model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Lambda
from keras import regularizers
from keras import backend as K

nb_var=hd.shape[1] - 1 #Number of variables


def create_base_network(input_shape):
    input = Input(shape =  input_shape)
    x = input
    x = Dense(512, 
              input_shape = (nb_var,), #Number of variables
              activation='relu', 
              name = 'D1')(x)
              #activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(512, 
              activation='relu', 
              name = 'D2')(x)
              #activity_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(512, 
              activation='relu',
              name = 'Embeddings')(x)
              #activity_regularizer = regularizers.l2(0.01))(x)
    return Model(input, x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis = 1, keepdims = True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print(shape1)
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

input_shape = (nb_var,) 
base_network = create_base_network(input_shape) #Verify what is the issue

input_a = Input(shape = input_shape)
input_b = Input(shape = input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, 
                  output_shape = eucl_dist_output_shape, 
                  name = 'Distance')([processed_a, processed_b])

model = Model([input_a, input_b], distance)

rms = RMSprop()

model.compile(loss = contrastive_loss, 
              optimizer = rms, 
              metrics=[accuracy])

model.summary()

#Training the network
y_train = tf.cast(y_train, tf.float32)
for i in range(1):
    history = model.fit([x_train[:, 0], x_train[:, 1]], y_train,
          batch_size = 256,
          epochs = 200,
          validation_split = 0.2,
          verbose = 1)

#Saving the model
save_path_model = path+'siamese_model_zero_shot.h5'
model.save(save_path_model)

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
np.save(path+'/embeddings_zero_shot_verify_train.npy', weights)

#######################
# Generate embeddings #
#######################

#Train set
hd = pd.read_csv(path+'spect_drugEmbFingerPrint_Resist_train.csv')
hd=hd.drop(['sample_id'], axis = 1) # Remove column name 'sample_id'
hd=hd.drop(['drug'], axis = 1) # Remove column name 'drug'
hd=hd.drop(['species'], axis = 1) # Remove column name 'species'
hd=hd.drop(['dataset'], axis = 1) # Remove column name 'dataset' (i.e. DRIAMS)
x = hd.iloc[:, 1:]
base_model = model.get_layer('model')
pred = base_model.predict(x)
np.save(path+'/embeddings_train.npy', pred)

#Test set
hd = pd.read_csv(path+'spect_drugEmbFingerPrint_Resist_test.csv')
hd=hd.drop(['sample_id'], axis = 1) # Remove column name 'sample_id'
hd=hd.drop(['drug'], axis = 1) # Remove column name 'drug'
hd=hd.drop(['species'], axis = 1) # Remove column name 'species'
hd=hd.drop(['dataset'], axis = 1) # Remove column name 'dataset' (i.e. DRIAMS)
x = hd.iloc[:, 1:]
pred = base_model.predict(x)
np.save(path+'/embeddings_test.npy', pred)

