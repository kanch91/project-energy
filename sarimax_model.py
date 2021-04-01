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

# For finding the best set of values by using a brute-force approach
p = d = q = range(0, 2)

# p, q, d values
pdq = list(iter.product(p, d, q))

# Seasonal P, Q and D values
seasonal_PQD = [(x[0], x[1], x[2], 2880) for x in list(iter.product(p, d, q))]

i = 0
AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_PQD:
        i += 1
        model = stats.tsa.statespace.SARIMAX(y_test, order=param, seasonal_order=param_seasonal,
                                           enforce_stationarity=False, enforce_invertibility=False)

        results = model.fit()

        print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
        AIC.append(results.aic)
        SARIMAX_model.append([param, param_seasonal])

print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],
                                                             SARIMAX_model[AIC.index(min(AIC))][1]))

# SARIMAX model
model = stats.tsa.statespace.SARIMAX(y_test[0:31925], order=[1, 1, 1],
                                   seasonal_order=[1,1,1,2880], enforce_stationarity=False,
                                   enforce_invertibility=False)


results = model.fit()
print(results.summary())

# results.plot_diagnostics()
# plt.show()

# y_pred = results.get_prediction(start=328, dynamic=True)
# pred_ci = y_pred.conf_int()

y_pred = results.get_forecast(steps=3546)

# Get confidence intervals of forecasts
pred_ci = y_pred.conf_int()

print(y_test[31925:], y_pred.predicted_mean)
# mse = mean_squared_error(y_test[327:]*100, y_pred*100)
y_forecasted = y_pred.predicted_mean
y_truth = y_test[31925:]


# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()

print("MSE: ", mse)
# print('RMSE:', mean_squared_error(y_test[327:] * 100, y_pred.predicted_mean*100, squared=False))
# print('R-squared:', r2_score(y_test[327:], y_pred.predicted_mean))

# Plotting the results
fig = plt.figure(figsize=(60, 8))
ax = y_test[31925:].plot(label='Observed')
y_pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
# ax.fill_betweenx(ax.get_ylim(), y_test.index[-1],
#                  alpha=.1, zorder=-1)
ax.set_xlabel('Daily')
ax.set_ylabel('Electricity Load')

# plt.plot(y_test[31925:]*100, label='Actual')
# plt.plot(y_pred*100, label='Predicted')
plt.legend(loc='upper right')
# plt.title("SARIMAX", fontsize=14)
# plt.xlabel('Daily')
# plt.ylabel('Electricity load')
plt.show()
plt.savefig('results/SARIMAX/final_output.jpg', bbox_inches='tight')

# Storing the result in a file: 'load_forecasting_result.txt'
predicted_test_result = y_pred * 100
np.savetxt('results/SARIMAX/predicted_values.txt', predicted_test_result)
actual_test_result = y_test[31925:] * 100
np.savetxt('results/SARIMAX/test_values.txt', actual_test_result)

