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

L1_num_params = 2 * 3 + 3 # weights + biases
L2_num_params = 3 * 1 + 1 # weights + biases
print("L1 paras = ", L1_num_params," L2 paras = ", L2_num_params)

W1,b1 = model.get_layer("layer_1").get_weights() # get weights and biases
W2,b2 = model.get_layer("layer_2").get_weights() # get weights and biases

print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:\n", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:\n", b2)


model.compile (
    loss =tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
)

model.fit(
    Xt,Yt,
    epochs = 10,
)

W1,b1 = model.get_layer("layer_1").get_weights() # get weights and biases
W2,b2 = model.get_layer("layer_2").get_weights() # get weights and biases
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:\n", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:\n", b2)


X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example

X_testn = norm_1(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decision = \n{yhat}")

yhat = (predictions >= 0.5).astype(int)
print(f"decision = \n{yhat}")

plt_layer(X,Y.reshape(-1), W1,b1,norm_1)
plt_output_unit(W2,b2)
netf = lambda x: model.predict(norm_1(x))
plt_network(X, Y, netf)