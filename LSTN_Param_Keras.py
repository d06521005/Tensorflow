'''
Resource code：　https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
Purpose: understand TimeDistributedDense
TimeDistributedDense applies a same Dense (fully-connected) 
operation to every timestep of a 3D tensor.
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization


###################################################
# <<   One-to-One LSTM for Sequence Prediction   >>

# Generating Input & otuput data
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(len(seq), 1, 1) # input data must 3-dim
y = seq.reshape(len(seq), 1) # output data must 2-dim
# X = Y = [0.0, 0.2, 0.4, 0.6, 0.8]

length = 5
a =  4*((1+1)*length + length**2) # calculate number of lSTM layer parameters
# imput:m, number of cell:n;  number of parameters = 4*(m+1 + n)*n
# m = input-dim + bias, n cell have n-dim of ht

# try understand the LSTM & Sequence
n_epoch = 1  # key 1 only for check parameter

# define LSTM configuration
n_neurons = length
n_batch = length

# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1))) # 5 LSTM cell as hiden layer
model.add(Dense(1)) # one neural to output
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())


# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)


# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
    print('%.1f' % value)
    
   
   
   
##############################################################################
# <<   Many-to-One LSTM for Sequence Prediction (without TimeDistributed)   >>

# Generating Input & otuput data
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(1, len(seq), 1) # input data must 3-dim
y = seq.reshape(1, len(seq)) # output data must 2-dim

length = len(seq)

# try understand the LSTM & Sequence
n_epoch = 1  # key 1 only for check parameter
n_neurons = length
n_batch = 1

# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1))) # 5 LSTM cell ; 5-dim input 
model.add(Dense(length)) # output layer with 5 neurals
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# five neural * 5 input + 5bias  = 30 (output layer parameters)


# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
    print('%.1f' % value)
    
    
    
############################################################################
# <<   Many-to-Many LSTM for Sequence Prediction (with TimeDistributed)   >>

# Generating Input & otuput data
length = 5
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)


# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 5

# create LSTM
model = Sequential()
model.add(LSTM(n_neurons,       # 5 LSTM cell
               input_shape=(length, 1), # 5-dim input 
               return_sequences=True))  # True: output at all steps. False: output as last step.
model.add(TimeDistributed(Dense(1)))
# use the TimeDistributed on the output layer to wrap a fully connected Dense layer with a single output
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# output layer only needs one connection to each LSTM unit (plus one bias)


# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
    print('%.1f' % value)
    
    
'''Purpose: understand TimeDistributedDense
TimeDistributedDense applies a same Dense (fully-connected) 
operation to every timestep of a 3D tensor.
End
'''






