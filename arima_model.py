import warnings
import pandas as pd
import numpy as np
import itertools as iter
import statsmodels.api as stats
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Pre-processing of the data
df_raw = pd.read_csv('assets/hourly_loaddata.csv', header=None, skiprows=1)  # loading raw data from the CSV
df_raw_array = df_raw.values  # numpy array
y_train = df_raw[2]/100
y_train = y_train[0:31925]
y_test = df_raw[2]/100
# y_test = y_test[31925:]

# For daily data
# for i in range(0, len(df_raw_array)):
# if (i%24) == 0:
#     y_test.append(df_raw[2].iloc[i:i+24].sum())


# y_test = np.array(y_test)
# print("y_test: ",y_test.shape,"\n", y_test, "\n")

# #For finding the value of d
# result = adfuller(y_test)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# #p<0.05 and so d = 0
#
#
# #For finding the value of p
# autocorrelation_plot(y_test)
# plt.show()
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = plot_acf(y_test.iloc[13:],lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = plot_pacf(y_test.iloc[13:],lags=40,ax=ax2)
#

# # For finding the best set of values by using a brute-force approach
# p = d = q = range(0, 4)
#
# # p, q, d values
# pdq = list(iter.product(p, d, q))
#
# i = 0
# AIC = []
# ARIMA_model = []
# for param in pdq:
#     i += 1
#     model = stats.tsa.arima.ARIMA(y_test, order=param,
#                                        enforce_stationarity=True, enforce_invertibility=True)
#
#     results = model.fit()
#
#     print('ARIMA: ', param, '\nAIC:',results.aic,'\n')
#     AIC.append(results.aic)
#     ARIMA_model.append([param])
#
# print('The smallest AIC is {} for model ARIMA{}'.format(min(AIC), ARIMA_model[AIC.index(min(AIC))][0],
#                                                              ARIMA_model[AIC.index(min(AIC))][0]))

# ARIMA model
model = stats.tsa.arima.ARIMA(y_test[0:31925], order=[3, 1, 3], enforce_stationarity=False,
                                   enforce_invertibility=False)


results = model.fit()
print(results.summary())


y_pred = results.predict(start=31925, end=35471, dynamic=True)
print(y_pred)


mse = mean_squared_error(y_test[31925:]*100, y_pred*100)
print("MSE: ", mse)
print('RMSE:', mean_squared_error(y_test[31925:] * 100, y_pred*100, squared=False))
print('R-squared:', r2_score(y_test[31925:], y_pred))

# Plotting the results
fig = plt.figure(figsize=(60, 8))
plt.plot(y_test[31925:]*100, label='Actual')
plt.plot(y_pred*100, label='Predicted')
plt.legend(loc='upper right')
plt.title("ARIMA", fontsize=14)
plt.xlabel('Hour')
plt.ylabel('Electricity load')
plt.show()
fig.savefig('results/ARIMA/final_output.jpg', bbox_inches='tight')

# Storing the result in a file: 'load_forecasting_result.txt'
predicted_test_result = y_pred * 100
np.savetxt('results/ARIMA/predicted_values.txt', predicted_test_result)
actual_test_result = y_test[31925:] * 100
np.savetxt('results/ARIMA/test_values.txt', actual_test_result)