from __future__ import print_function
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Activation
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import os

from tensorflow.python.keras.callbacks import EarlyStopping

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()

# Reading from the database
dataset = pd.read_csv('assets/hourly_load&weather_data.csv')
x_temperature = dataset['Temperature']
x_dewpoint = dataset['Dew Point']
x_humidity = dataset['Humidity']
x_windspeed = dataset['Wind Speed']
x_pressure = dataset['Pressure']
y = dataset['Load']

# Converting the values in the usable format
x_temperature = x_temperature.values
x_dewpoint = x_dewpoint.values
x_humidity = x_humidity.values
x_windspeed = x_windspeed.values
x_pressure = (x_pressure.values) / 29.53  # Hg to bar
y = (y.values)/1000 # MW to GW


# Visualising the independent variables
dataset_graph = plt.figure(figsize=(60, 8))
plt.plot(x_temperature[:364], label='Temperature (C)')
plt.plot(x_dewpoint[:364], label='Dew Point (C)')
plt.plot(x_humidity[:364], label='Humidity (%)')
plt.plot(x_windspeed[:364], label='Wind Speed (mph)')
plt.plot(x_pressure[:364], label='Pressure (bar)')
plt.plot(y[:364], label='Load (GW)')
plt.legend(loc='upper right')
plt.title("Dataset", fontsize=14)
plt.xlabel('Day', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend()
plt.show()
dataset_graph.savefig('results/GRU_weather/dataset_graph.jpg', bbox_inches='tight')


# Converting to a usable format in a 2D array
x_temperature = x_temperature.reshape((len(x_temperature), 1))
x_dewpoint = x_dewpoint.reshape((len(x_dewpoint), 1))
x_humidity = x_humidity.reshape((len(x_humidity), 1))
x_windspeed = x_windspeed.reshape((len(x_windspeed), 1))
x_pressure = x_pressure.reshape((len(x_pressure), 1))
y = y.reshape((len(y), 1))

print("\nIndependent & Dependent variables arrays:")
print("x_temperature:", x_temperature.shape)
print("x_dewpoint:", x_dewpoint.shape)
print("x_humidity:", x_humidity.shape)
print("x_windspeed:", x_windspeed.shape)
print("x_pressure:", x_pressure.shape)
print("y:", y.shape)

# Normalising the data using the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_temperature_scaled = scaler.fit_transform(x_temperature)
x_dewpoint_scaled = scaler.fit_transform(x_dewpoint)
x_humidity_scaled = scaler.fit_transform(x_humidity)
x_windspeed_scaled = scaler.fit_transform(x_windspeed)
x_pressure_scaled = scaler.fit_transform(x_pressure)
y_scaled = scaler.fit_transform(y)


# Stacking the data columns horizontally
stacked_dataset = np.hstack(
    (x_temperature_scaled, x_dewpoint_scaled, x_humidity_scaled, x_windspeed_scaled, x_pressure_scaled, y_scaled))
print("\nStacked Dataset", stacked_dataset.shape, ":\n", stacked_dataset)


#Split a multivariate sequence into samples
def split_sequences(dataset, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(dataset)):
        start = i + n_steps_in
        end = start + n_steps_out-1
        if end > len(dataset):
            break
        x_sequence, y_sequence = dataset[i:start, :-1], dataset[start-1:end, -1]
        X.append(x_sequence)
        y.append(y_sequence)
    return np.array(X), np.array(y)

n_steps_in, n_steps_out, n_features = 30 , 15, 5
X, y = split_sequences(stacked_dataset, n_steps_in, n_steps_out)
print ("\nX.shape " , X.shape)
print ("y.shape" , y.shape)

#Splitting into training & test sets
train_X , train_y = X[:255, :] , y[:255, :]
test_X , test_y = X[255:, :] , y[255:, :]
print ("\ntrain_X" , train_X.shape)
print ("train_y" , train_y.shape)
print ("test_X" , test_X.shape)
print ("test_y" , test_y.shape)


#Learning Rate
opt = keras.optimizers.Adam(learning_rate=0.001)

#GRU Model
model = Sequential()
model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(GRU(50, activation='relu'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.compile(loss='mse' , optimizer=opt , metrics=['mse'])
es = EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto", baseline=None,
                   restore_best_weights=True)  # Stops the training when the values don't improve
history = model.fit(train_X, train_y, epochs=100, batch_size=6, verbose=1, validation_data=(test_X, test_y), callbacks=[es], shuffle=False)
print(model.summary())

# Evaluating the result
test_mse = model.evaluate(test_X, test_y, verbose=1)
# print('\nThe Mean-squared-error (MSE) on the test data set is %.6f over %d test samples.' % (test_mse, len(test_y)))
print("Test MSE:", test_mse)

# print("X", test_X.shape, test_X[65])
# Getting the predicted values
predicted_values = model.predict(test_X[0].reshape((1,30,5)))
# scaler1 = MinMaxScaler(feature_range=(0, 1))
# scaler1.fit(y)
# y_pred = scaler1.inverse_transform(predicted_values)
# print("Y", y_pred)
print("P", predicted_values, predicted_values.reshape((15,)))
print("y", test_y[0], test_y[0].shape)

# # read test data
# x_temperature = dataset['Temperature'].values
# x_dewpoint = dataset['Dew Point'].values
# x_humidity = dataset['Humidity'].values
# x_windspeed = dataset['Wind Speed'].values
# x_pressure = dataset['Pressure'].values
# y_test = dataset['Load'].values
#
# x_temperature = x_temperature[255:]
# x_dewpoint = x_dewpoint[255:]
# x_humidity = x_humidity[255:]
# x_windspeed = x_windspeed[255:]
# x_pressure = x_pressure[255:]
# y_test = y_test[255:]
#
#
# # convert to [rows, columns] structure
# x_temperature = x_temperature.reshape((len(x_temperature), 1))
# x_dewpoint = x_dewpoint.reshape((len(x_dewpoint), 1))
# x_humidity = x_humidity.reshape((len(x_humidity), 1))
# x_windspeed = x_windspeed.reshape((len(x_windspeed), 1))
# x_pressure = x_pressure.reshape((len(x_pressure), 1))
# y_test = y_test.reshape((len(y_test), 1))
#
# x_temperature_scaled = scaler.fit_transform(x_temperature)
# x_dewpoint_scaled = scaler.fit_transform(x_dewpoint)
# x_humidity_scaled = scaler.fit_transform(x_humidity)
# x_windspeed_scaled = scaler.fit_transform(x_windspeed)
# x_pressure_scaled = scaler.fit_transform(x_pressure)
#
#
# def prep_data(x_temperature_scaled, x_dewpoint_scaled, x_humidity_scaled, x_windspeed_scaled, x_pressure_scaled, y_test, start, end, last):
#     dataset_test = np.hstack((x_temperature_scaled, x_dewpoint_scaled, x_humidity_scaled, x_windspeed_scaled, x_pressure_scaled))
#     dataset_test_X = dataset_test[start:end, :]
#     test_X_new = dataset_test_X.reshape(1, dataset_test_X.shape[0], dataset_test_X.shape[1])
#
#     # prepare past and groundtruth
#     past_data = y_test[:end, :]
#     dataset_test_y = y_test[end:last, :]
#     scaler1 = MinMaxScaler(feature_range=(0, 1))
#     scaler1.fit(dataset_test_y)
#
#     # predictions
#     y_pred = model.predict(test_X_new)
#     y_pred_inv = scaler1.inverse_transform(y_pred)
#     y_pred_inv = y_pred_inv.reshape(n_steps_out, 1)
#     y_pred_inv = y_pred_inv[:, 0]
#
#     return y_pred_inv, dataset_test_y, past_data
#
#
# def evaluate_prediction(predictions, actual, model_name , start , end):
#     errors = predictions - actual
#     mse = np.square(errors).mean()
#     rmse = np.sqrt(mse)
#     mae = np.abs(errors).mean()
#
#     print("Test Data from {} to {}".format(start, end))
#     print('Mean Absolute Error: {:.2f}'.format(mae))
#     print('Root Mean Square Error: {:.2f}'.format(rmse))
#     print('')
#     print('')
#
#
#
# # Plot history and future
# def plot_multistep(history, prediction1 , groundtruth , start , end):
#     plt.figure(figsize=(20, 4))
#     y_mean = np.mean(prediction1)
#     range_history = len(history)
#     range_future = list(range(range_history, range_history + len(prediction1)))
#     plt.plot(np.arange(range_history), np.array(history), label='History')
#     plt.plot(range_future, np.array(prediction1),label='Forecasted with GRU')
#     plt.plot(range_future, np.array(groundtruth),label='GroundTruth')
#     plt.legend(loc='upper right')
#     plt.title("Test Data from {} to {} , Mean = {:.2f}".format(start, end, y_mean) ,  fontsize=18)
#     plt.xlabel('Time step' ,  fontsize=18)
#     plt.ylabel('y-value' , fontsize=18)
#
#
# for i in range(30, 60, 90):
#     start = i
#     end = start + n_steps_in
#     last = end + n_steps_out
#     y_pred_inv , dataset_test_y , past_data = prep_data(x_temperature_scaled, x_dewpoint_scaled, x_humidity_scaled, x_windspeed_scaled, x_pressure_scaled, y_test, start, end, last)
#     evaluate_prediction(y_pred_inv , dataset_test_y, 'GRU' , start , end)
#     plot_multistep(past_data , y_pred_inv , dataset_test_y , start , end)

# Plotting the results
fig = plt.figure()
plt.plot((test_y[0]))
plt.plot(predicted_values.reshape((15,)))
plt.xlabel('Days')
plt.ylabel('Electricity load (*1e2)')
plt.legend(('Actual', 'Predicted'), fontsize='15')
plt.show()
fig.savefig('results/GRU_weather/final_output.jpg', bbox_inches='tight')

# Plot of the loss
loss_fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
loss_fig.savefig('results/GRU_weather/final_loss.jpg', bbox_inches='tight')

# Storing the result in a file: 'load_forecasting_result.txt'
np.savetxt('results/GRU_weather/predicted_values.txt', predicted_values)
np.savetxt('results/GRU_weather/test_values.txt', test_y)

end_time = time.time()

# print("MSE:", mean_squared_error(test_y[65].reshape((1,15)), y_pred))
print("MSE:", mean_squared_error(test_y[0].reshape((1,15)), predicted_values))
print("Total time: ", end_time - start_time)
