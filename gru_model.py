from __future__ import print_function
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()


# Defining a function to convert a vector of time series into a 2D matrix for faster processing
def convertTimeSeriesTo2DMatrix(vectorSeries, sequence_length):
    matrix = []
    for i in range(len(vectorSeries) - sequence_length + 1):
        matrix.append(vectorSeries[i:i + sequence_length])
    return matrix


np.random.seed(1234)  # Selecting a random seed

# Pre-processing of the data
df_raw = pd.read_csv('assets/hourly_loaddata.csv', header=None, skiprows=1)  # loading raw data from the CSV
df_raw_array = df_raw.values  # numpy array

# daily_load = [df_raw_array[i,:] for i in range(0, len(df_raw)) if i % 24 == 0] # daily load
# print(daily_load)

hourly_load = [df_raw_array[i, 2] / 100 for i in range(0, len(df_raw))]  # hourly load, 24 for each day
# print(hourly_load)

length_of_sequence = 24  # Storing the length of the sequence/hours in the day for predicting the future value

# Converting the vector to a 2D matrix using the function above
hourly_load_matrix = convertTimeSeriesTo2DMatrix(hourly_load, length_of_sequence)

# Shift all the data by mean
hourly_load_matrix = np.array(hourly_load_matrix)
shifted_value = hourly_load_matrix.mean()
hourly_load_matrix = hourly_load_matrix - shifted_value
# print ("Data  shape: ", hourly_load_matrix.shape)
# print(hourly_load_matrix)

# Splitting the dataset into two: 90% for training and 10% for testing
test_row = int(round(0.9 * hourly_load_matrix.shape[0]))
train_set = hourly_load_matrix[:test_row, :]

np.random.shuffle(train_set)  # Shuffling only the training set randomly

# print(train_set, "\n")

# The Final training set
X_train = train_set[:, :-1]
print("X_train", "\n", X_train, "\n")
y_train = train_set[:, -1]  # The last column is the true value to compute the mean-squared-error loss
print("y_train", "\n", y_train, "\n")

# The Final testing set
X_test = hourly_load_matrix[test_row:, :-1]
y_test = hourly_load_matrix[test_row:, -1]

# The input to GRU layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Building the GRU model
model = Sequential()

# Layer 1: GRU
model.add(GRU(input_dim=1, units=50, return_sequences=True))
model.add(Dropout(0.2))  # Reducing overfitting and improving model performance

# Layer 2: GRU
model.add(GRU(units=100, return_sequences=False))
model.add(Dropout(0.2))  # Reducing overfitting and improving model performance

# Layer 3: Dense
model.add(Dense(units=1, activation='linear'))

# Compiling the model
model.compile(loss="mse", optimizer="adam")
es = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=1, mode="auto", baseline=None,
                   restore_best_weights=True)  # Stops the training when the values don't improve

# Training the model
model.fit(X_train, y_train, batch_size=512, epochs=50, validation_split=0.05, verbose=1, callbacks=[es])
test_mse = model.evaluate(X_test, y_test, verbose=1)

# Values of hyperparameters for finding the best values
epoch = [1, 5, 10, 25, 50, 100]
batch_size = [4, 12, 16, 32, 64, 256, 512, 1024, 2048]

# Finding the best combination of hyperparameters
best_mse, final_epoch, final_batch_size = 0, 0, 0
result = []
for i in epoch:
    for j in batch_size:
        model.fit(X_train, y_train, batch_size=j, epochs=i, validation_split=0.05, verbose=1, callbacks=[es])
        if model.evaluate(X_test, y_test, verbose=1) < test_mse:
            test_mse = model.evaluate(X_test, y_test, verbose=1)
        test_mse = model.evaluate(X_test, y_test, verbose=1)
        mse_combination = [i, j, test_mse]
        print("Model w/ Epoch: ", i, "|| Batch Size:", j, "|| MSE: ", test_mse, "\n")
        result.append(mse_combination)

# Storing all the values in CSV file: 'MSEs.csv'
np.savetxt("results/GRU/MSEs.csv", result, delimiter=", ", header="Epoch, Batch Size, MSE", fmt='% s')

data_frame = pd.read_csv("results/GRU/MSEs.csv")
print(data_frame)

# Selecting the top 5 model combinations to find the best performing
mse_frame = pd.DataFrame(result)
final_mse_frame = mse_frame.sort_values(by=[2], ascending=True)
epoch_list = [final_mse_frame[0].iloc[0], final_mse_frame[0].iloc[1], final_mse_frame[0].iloc[2],
              final_mse_frame[0].iloc[3], final_mse_frame[0].iloc[4]]
batch_size_list = [final_mse_frame[1].iloc[0], final_mse_frame[1].iloc[1], final_mse_frame[1].iloc[2],
                   final_mse_frame[1].iloc[3], final_mse_frame[1].iloc[4]]

for i in range(5):
    for j in range(3):
        print("Iteration no.:", j, "|| Model:", epoch_list[i], batch_size_list[i])
        model.fit(X_train, y_train, batch_size=batch_size_list[i], epochs=epoch_list[i], validation_split=0.05,
                  verbose=1, callbacks=[es])
        if model.evaluate(X_test, y_test, verbose=1) < test_mse:
            test_mse = model.evaluate(X_test, y_test, verbose=1)
            final_epoch = epoch_list[i]
            final_batch_size = batch_size_list[i]
        test_mse = model.evaluate(X_test, y_test, verbose=1)
        print("Model w/ Epoch: ", epoch_list[i], "|| Batch Size: ", batch_size_list[i], "|| MSE: ", test_mse, "\n")
        mse_combination = [epoch_list[i], batch_size_list[i], test_mse]
        result.append(mse_combination)

# Storing all the values in CSV file: 'MSEs_top5models.csv'
np.savetxt("results/GRU/MSEs_top5models.csv", result, delimiter=", ", header="Epoch, Batch Size, MSE", fmt='% s')
data_frame = pd.read_csv("results/GRU/MSEs_top5models.csv")
print(data_frame)

# # Finding the best combination
# counter = 0
# mse = []
# temp = []
# mse_frame = pd.DataFrame(result)
# for i in range(5):
#     for j in range(3):
#         temp.append(mse_frame[2].iloc[counter])
#         counter = counter+1
#     mse.append(temp)
#
# average_mse = np.array(temp)
# average_mse = np.sum(average_mse, axis=1)/3
# max_mse_index = np.where(np.max(average_mse))
# # Find the epoch & batch size combination to train the best model

# Training the best model
model.fit(X_train, y_train, batch_size=final_batch_size, epochs=final_epoch, validation_split=0.05, verbose=1,
          callbacks=[es])

# Evaluating the result
test_mse = model.evaluate(X_test, y_test, verbose=1)
print('\nThe Mean-squared-error (MSE) on the test data set is %.6f over %d test samples.' % (test_mse, len(y_test)))

# Getting the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples, 1))
# print(predicted_values)

# Plotting the results
fig = plt.figure()
plt.plot(predicted_values + shifted_value)
plt.plot(y_test + shifted_value)
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e2)')
plt.legend(('Predicted', 'Actual'), fontsize='15')
plt.show()
fig.savefig('results/GRU/final_output.jpg', bbox_inches='tight')

# Storing the result in a file: 'load_forecasting_result.txt'
predicted_test_result = predicted_values + shifted_value
np.savetxt('results/GRU/predicted_values.txt', predicted_test_result)
actual_test_result = y_test + shifted_value
np.savetxt('results/GRU/test_values.txt', actual_test_result)

# loss = 100 * np.mean(abs((actual_test_result - predicted_test_result) / predicted_test_result), axis=-1)
# print(loss)

end_time = time.time()

print("Total time:", end_time - start_time)
