from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Defining a function to convert a vector of time series into a 2D matrix for faster processing
def convertTimeSeriesTo2DMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

np.random.seed(1234) # Selecting a random seed

# Pre-processing of the data
df_raw = pd.read_csv('hourly_loaddata.csv', header=None, skiprows=1) # loading raw data from the CSV
df_raw_array = df_raw.values # numpy array

# daily_load = [df_raw_array[i,:] for i in range(0, len(df_raw)) if i % 24 == 0] # daily load
# print(daily_load)

hourly_load = [df_raw_array[i,2]/100 for i in range(0, len(df_raw))] # hourly load, 24 for each day
print(hourly_load)

length_of_sequence = 24 # Storing the length of the sequence for predicting the future value

# Converting the vector to a 2D matrix using the function above
hourly_load_matrix = convertTimeSeriesTo2DMatrix(hourly_load, length_of_sequence)

# Shift all the data by mean
hourly_load_matrix = np.array(hourly_load_matrix)
shifted_value = hourly_load_matrix.mean()
hourly_load_matrix = hourly_load_matrix - shifted_value
print ("Data  shape: ", hourly_load_matrix.shape)
# print(hourly_load_matrix)

# Splitting the dataset into two: 90% for training and 10% for testing
test_row = int(round(0.9 * hourly_load_matrix.shape[0]))
train_set = hourly_load_matrix[:test_row, :]

np.random.shuffle(train_set) # Shuffling randomly only the training set

print(train_set, "\n")

# The Final training set
X_train = train_set[:, :-1]
print(X_train, "\n")
y_train = train_set[:, -1] # The last column is the true value to compute the mean-squared-error loss
print(y_train)

# The Final testing set
X_test = hourly_load_matrix[test_row:, :-1]
y_test = hourly_load_matrix[test_row:, -1]

# The input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# Building the LSTM model
model = Sequential()

# Layer 1: LSTM
model.add(LSTM(input_dim=1, units=50, return_sequences=True))
model.add(Dropout(0.2)) # Reducing overfitting and improving model performance

# Layer 2: LSTM
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2)) # Reducing overfitting and improving model performance

# Layer 3: Dense
model.add(Dense(units=1, activation='linear'))

# Compiling the model
model.compile(loss="mse", optimizer="rmsprop")

# Training the model
model.fit(X_train, y_train, batch_size=512, epochs=50, validation_split=0.05, verbose=1)

# Evaluating the result
test_mse = model.evaluate(X_test, y_test, verbose=1)
print ('\nThe Mean-squared-error (MSE) on the test data set is %.4f over %d test samples.' % (test_mse, len(y_test)))

# Getting the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))
# print(predicted_values)

# Plotting the results
fig = plt.figure()
plt.plot(predicted_values + shifted_value)
plt.plot(y_test + shifted_value)
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e2)')
plt.show()
fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')

# Storing the result in a file: 'load_forecasting_result.txt'
test_result = np.vstack((predicted_values, y_test)) + shifted_value
np.savetxt('load_forecasting_result.txt', test_result)

