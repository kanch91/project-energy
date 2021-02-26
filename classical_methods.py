import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# Defining a function to convert a vector of time series into a 2D matrix for faster processing
def convertTimeSeriesTo2DMatrix(vectorSeries, sequence_length):
    matrix = []
    for i in range(len(vectorSeries) - sequence_length + 1):
        matrix.append(vectorSeries[i:i + sequence_length])
    return matrix


# Pre-processing of the data
df_raw = pd.read_csv('assets/hourly_loaddata.csv', header=None, skiprows=1)  # loading raw data from the CSV
df_raw_array = df_raw.values  # numpy array

hourly_load = [df_raw_array[i, 2] / 100 for i in range(0, len(df_raw))]  # hourly load, 24 for each day

length_of_sequence = 24  # Storing the length of the sequence/hours in the day for predicting the future value

# Converting the vector to a 2D matrix using the function above
hourly_load_matrix = convertTimeSeriesTo2DMatrix(hourly_load, length_of_sequence)

# Shift all the data by mean
hourly_load_matrix = np.array(hourly_load_matrix)
hourly_load_matrix = hourly_load_matrix

# Splitting the dataset and using only the 10% used for testing in other models
test_row = int(round(0.9 * hourly_load_matrix.shape[0]))

# The Final testing set
y_test = hourly_load_matrix[test_row:, -1]
print("y_test: ", y_test.shape, "\n", y_test, "\n")


def simple_moving_average(n, y_test):
    # Getting the predicted values for SMA
    y_pred = pd.Series(y_test).rolling(window=n).mean().iloc[n - 1:].values
    print("Predicted values: ", y_pred, "\n")
    mse_sma = mean_squared_error(y_test[n - 1:], y_pred)

    # Plotting the results
    fig = plt.figure(figsize=(60, 8))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend(loc='upper right')
    plt.title("Simple Moving Average", fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load')
    plt.show()
    fig.savefig('results/SMA/final_output.jpg', bbox_inches='tight')

    # Storing the result in a file: 'load_forecasting_result.txt'
    predicted_test_result = y_pred
    np.savetxt('results/SMA/predicted_values.txt', predicted_test_result)
    actual_test_result = y_test
    np.savetxt('results/SMA/test_values.txt', actual_test_result)

    return mse_sma, y_pred


def weighted_moving_average(n, y_test):
    y_pred = []
    for i in range(len(y_test) - 4):
        total = np.arange(1, n + 1, 1)  # Weight Matrix for current being the heaviest
        temp = y_test[i:i + n]
        temp = total * temp
        wma = (temp.sum()) / (total.sum())
        y_pred = np.append(y_pred, wma)

    print("Predicted values: ", y_pred, "\n")
    mse_wma = mean_squared_error(y_test[n - 1:], y_pred)

    # Plotting the results
    fig = plt.figure(figsize=(60, 8))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend(loc='upper right')
    plt.title("Weighted Moving Average", fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load')
    plt.show()
    fig.savefig('results/WMA/final_output.jpg', bbox_inches='tight')

    # Storing the result in a file: 'load_forecasting_result.txt'
    predicted_test_result = y_pred
    np.savetxt('results/WMA/predicted_values.txt', predicted_test_result)
    actual_test_result = y_test
    np.savetxt('results/WMA/test_values.txt', actual_test_result)

    return mse_wma, y_pred


def cumulative_moving_average(y_test):
    df = pd.DataFrame(y_test)
    y_pred = df.expanding().mean()

    print("Predicted values: ", y_pred, "\n")
    mse_cma = mean_squared_error(y_test, y_pred)

    # Plotting the results
    fig = plt.figure(figsize=(60, 8))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend(loc='upper right')
    plt.title("Cumulative Moving Average", fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load')
    plt.show()
    fig.savefig('results/CMA/final_output.jpg', bbox_inches='tight')

    # Storing the result in a file: 'load_forecasting_result.txt'
    predicted_test_result = y_pred
    np.savetxt('results/CMA/predicted_values.txt', predicted_test_result)
    actual_test_result = y_test
    np.savetxt('results/CMA/test_values.txt', actual_test_result)

    return mse_cma, y_pred


def exponential_moving_average(y_test):
    df = pd.DataFrame(y_test)
    smoothing_factor = 0.5
    y_pred = df.ewm(alpha=smoothing_factor, adjust=False).mean()

    print("Predicted values: ", y_pred, "\n")
    mse_ema = mean_squared_error(y_test, y_pred)

    # Plotting the results
    fig = plt.figure(figsize=(60, 8))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend(loc='upper right')
    plt.title("Exponential Moving Average", fontsize=14)
    plt.xlabel('Hour')
    plt.ylabel('Electricity load')
    plt.show()
    fig.savefig('results/EMA/final_output.jpg', bbox_inches='tight')

    # Storing the result in a file: 'load_forecasting_result.txt'
    predicted_test_result = y_pred
    np.savetxt('results/EMA/predicted_values.txt', predicted_test_result)
    actual_test_result = y_test
    np.savetxt('results/EMA/test_values.txt', actual_test_result)

    return mse_ema, y_pred


print("---------------------------------------------------------")

n = 5  # Window size
mse_sma, y_sma = simple_moving_average(n, y_test)
print("MSE for SMA: ", mse_sma)
print('RMSE for SMA:', mean_squared_error(y_sma, y_test[n - 1:], squared=False))
print('R-squared for SMA:', r2_score(y_sma, y_test[n - 1:]))

print("---------------------------------------------------------")

mse_cma, y_cma = cumulative_moving_average(y_test)
print("MSE for CMA: ", mse_cma)
print('RMSE for CMA:', mean_squared_error(y_cma, y_test, squared=False))
print('R-squared for CMA:', r2_score(y_cma, y_test))

print("---------------------------------------------------------")

mse_ema, y_ema = exponential_moving_average(y_test)
print("MSE for EMA: ", mse_ema)
print('RMSE for EMA:', mean_squared_error(y_ema, y_test, squared=False))
print('R-squared for EMA:', r2_score(y_ema, y_test))

print("---------------------------------------------------------")

mse_wma, y_wma = weighted_moving_average(n, y_test)
print("MSE for WMA: ", mse_wma)
print('RMSE for WMA:', mean_squared_error(y_wma, y_test[n - 1:], squared=False))
print('R-squared for WMA:', r2_score(y_wma, y_test[n - 1:]))

print("---------------------------------------------------------")

# Plotting the results
fig = plt.figure(figsize=(60, 8))
plt.plot(y_sma, label='SMA')
plt.plot(y_cma, label='CMA')
plt.plot(y_ema, label='EMA')
plt.plot(y_wma, label='WMA')
plt.plot(y_test, label='Actual Values')
plt.legend(loc='upper right')
plt.xlabel('Hour')
plt.ylabel('Electricity load')
plt.title("Predicted Values of various traditional methods", fontsize=14)
plt.show()
fig.savefig('results/Classical_final_output.jpg', bbox_inches='tight')
