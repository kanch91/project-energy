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

warnings.filterwarnings("ignore")

# Pre-processing of the data
df_raw = pd.read_csv('assets/hourly_loaddata.csv', header=None, skiprows=1)  # loading raw data from the CSV
df_raw_array = df_raw.values  # numpy array
y_test = df_raw[2]/100
y_test = y_test[31925:]
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

p = d = q = range(0, 2)

# p, q, d values
pdq = list(iter.product(p, d, q))

# Seasonal P, Q and D values
seasonal_PQD = [(x[0], x[1], x[2], 24) for x in list(iter.product(p, d, q))]

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

# Let's fit this model
model = stats.tsa.statespace.SARIMAX(y_test, order=[1, 0, 1],
                                   seasonal_order=[1,1,1,24], enforce_stationarity=False,
                                   enforce_invertibility=False)

results = model.fit()
print(results.summary())


y_pred = results.predict(y_test)
print(y_pred)