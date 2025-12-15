import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


X,Y = load_coffee_data();
print(X.shape, Y.shape)

#plt_roast(X,Y)

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}") # [;,0] is temperature column
print(f"Duration Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}") # [;,1] is duration column

norm_1 = tf.keras.layers.Normalization(axis=-1) # create a normalization layer
norm_1.adapt(X) # fit the state of the layer to the data
Xn = norm_1(X) # normalize the data 

print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}") # [;,0] is temperature column
print(f"Duration Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}") # [;,1] is duration column

Xt = np.tile(Xn,(1000,1)) # add a column of ones for the bias term
Yt = np.tile(Y,(1000,1)) # add a column of ones for the bias term
print(Xt.shape, Yt.shape)


#TensorFlow Sequential API

tf.random.set_seed(1234) # for reproducibility
model = Sequential(
    [
        tf.keras.Input(shape=(2,)), # input layer
        Dense(3, activation = 'sigmoid', name = 'layer_1'),
        Dense(1 , activation = 'sigmoid', name = 'layer_2') # hidden layer
    ]
)

model.summary() # print the model summary