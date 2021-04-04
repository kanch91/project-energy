import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# Pre-processing of the data
df_raw = pd.read_csv('assets/hourly_load&weather_data.csv', header=None, skiprows=1)  # loading raw data from the CSV
df_raw_array = df_raw.values  # numpy array
y_test = df_raw[1]/1000
x_test = np.delete(df_raw_array, 0, 1)
x_test = np.delete(x_test, 6, 1)
# print(x_test)
y_test = np.array(y_test)
# print("y_test: ", y_test.shape, "\n", y_test, "\n")


def single_exponential_smoothing(alpha, y_test):
    # Accounts only for the level of the series

    # Simple Exponential Smoothing
    ses_model1 = SimpleExpSmoothing(endog=y_test).fit(smoothing_level=alpha, optimized=True)
    y_pred_ses = ses_model1.predict(328).rename(r'$\alpha=%s$' % ses_model1.model.params['smoothing_level'])

    fig = plt.figure(figsize=(60, 8))
    y_pred_ses[1:].plot(color='grey', legend=True)
    ses_model1.fittedvalues.plot(color='grey')
    plt.title("Single Exponential Smoothing")
    plt.show()
    fig.savefig('results/SES_weather/final_output.jpg', bbox_inches='tight')

    # print("Predicted values: ", y_pred, "\n")
    mse_ses = mean_squared_error(y_test[327:-1], y_pred_ses)
    rmse_ses = mean_squared_error(y_pred_ses, y_test[327:-1], squared=False)
    r2_ses = r2_score(y_test[327:-1], y_pred_ses)

    # Storing the result in a file: 'load_forecasting_result.txt'
    predicted_test_result = y_pred_ses
    np.savetxt('results/SES_weather/predicted_values.txt', predicted_test_result)
    actual_test_result = y_test
    np.savetxt('results/SES_weather/test_values.txt', actual_test_result)

    return mse_ses, rmse_ses, r2_ses, y_pred_ses


def double_exponential_smoothing(alpha, beta, y_test):
    # Accounts for level + trend in the data

    des_model = Holt(y_test).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    y_pred_des = des_model.predict(328).rename("Holt's Linear")

    fig = plt.figure(figsize=(60, 8))
    des_model.fittedvalues.plot(color='grey')
    y_pred_des.plot(color='grey', legend=True)
    fig.savefig('results/DES_weather/final_output.jpg', bbox_inches='tight')
    plt.title("Holt's Method/Double Exponential Smoothing")
    plt.show()

    # print("Predicted values: ", y_pred_des, "\n")
    mse_des = mean_squared_error(y_test[327:-1], y_pred_des)
    rmse_des = mean_squared_error(y_pred_des, y_test[327:-1], squared=False)
    r2_des = r2_score(y_test[327:-1], y_pred_des)

    np.savetxt('results/DES_weather/predicted_values_model.txt', y_pred_des)
    actual_test_result = y_test
    np.savetxt('results/DES_weather/test_values.txt', actual_test_result)

    # Three models with different parameters
    # #Model 1: Providing the model with the values of hyperparameters (alpha, beta)
    # des_model1 = Holt(y_test).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    # y_pred_des1 = des_model1.predict(31924).rename("Holt's Linear")
    #
    # #Model 2: Exponential Model with same alpha & beta
    # des_model2 = Holt(y_test, exponential=True).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    # y_pred_des2 = des_model2.predict(31924).rename("Exponential")
    #
    # #Model 3: Optimising the dampening parameter with same alpha & beta
    # des_model3 = Holt(y_test, damped_trend=True).fit(smoothing_level=alpha, smoothing_trend=beta)
    # y_pred_des3 = des_model3.predict(31924).rename("Additive damped trend")
    #
    # fig = plt.figure(figsize=(60, 8))
    # des_model1.fittedvalues.plot(color='blue')
    # y_pred_des1.plot(color='blue', legend=True)
    # des_model2.fittedvalues.plot(color='red')
    # y_pred_des2.plot(color='red', legend=True)
    # des_model3.fittedvalues.plot(color='green')
    # y_pred_des3.plot(color='green', legend=True)
    # fig.savefig('results/DES/final_output.jpg', bbox_inches='tight')
    # plt.title("Holt's Method/Double Exponential Smoothing")
    # plt.show()
    #
    # print("Predicted values (Model 1): ", y_pred_des1, "\n")
    # print("Predicted values (Model 2): ", y_pred_des2, "\n")
    # print("Predicted values (Model 3): ", y_pred_des3, "\n")
    # mse_des1 = mean_squared_error(y_test[31923:-1], y_pred_des1)
    # mse_des2 = mean_squared_error(y_test[31923:-1], y_pred_des2)
    # mse_des3 = mean_squared_error(y_test[31923:-1], y_pred_des3)
    #
    # np.savetxt('results/DES/predicted_values_model1.txt', y_pred_des1)
    # np.savetxt('results/DES/predicted_values_model2.txt', y_pred_des2)
    # np.savetxt('results/DES/predicted_values_model3.txt', y_pred_des3)
    # actual_test_result = y_test
    # np.savetxt('results/DES/test_values.txt', actual_test_result)
    #
    # return mse_des1, mse_des2, mse_des3

    return mse_des, rmse_des, r2_des, y_pred_des


def triple_exponential_smoothing(season, y_test):
    # Accounts for level + trend + seasonality in the data
    # Three models with different parameters

    # Model 1: Additive trend + season with box-cox transformation
    tes_model = ExponentialSmoothing(y_test, seasonal_periods=season, trend='add', seasonal='add').fit(use_boxcox=True)
    y_pred_tes = tes_model.predict(328).rename("TES")

    fig = plt.figure(figsize=(60, 8))
    tes_model.fittedvalues.plot(color='grey')
    y_pred_tes.plot(color='grey', legend=True)
    fig.savefig('results/TES_weather/final_output.jpg', bbox_inches='tight')
    plt.title("Holt-Winters' Method/Triple Exponential Smoothing")
    plt.show()

    # print("Predicted values: ", y_pred_tes, "\n")
    mse_tes = mean_squared_error(y_test[327:-1], y_pred_tes)
    rmse_tes = mean_squared_error(y_pred_tes, y_test[327:-1], squared=False)
    r2_tes = r2_score(y_test[327:-1], y_pred_tes)

    np.savetxt('results/TES_weather/predicted_values_model.txt', y_pred_tes)
    actual_test_result = y_test
    np.savetxt('results/TES_weather/test_values.txt', actual_test_result)

    # #Three models with different parameters
    #
    # #Model 1: Additive trend + season with box-cox transformation
    # tes_model1 = ExponentialSmoothing(y_test, seasonal_periods=season, trend='add', seasonal='add').fit(use_boxcox=True)
    # y_pred_tes1 = tes_model1.predict(31924).rename("Model 1")
    #
    # #Model 2: Additive trend + Multiplicative season with box-cox transformation
    # tes_model2 = ExponentialSmoothing(y_test, seasonal_periods=season, trend='add', seasonal='mul').fit(use_boxcox=True)
    # y_pred_tes2 = tes_model2.predict(31924).rename("Model 2")
    #
    # #Model 3: Damped trend + Additive season with box-cox transformation
    # tes_model3 = ExponentialSmoothing(y_test, seasonal_periods=season, trend='add', seasonal='add', damped_trend=True).fit(use_boxcox=True)
    # y_pred_tes3 = tes_model3.predict(31924).rename("Model 3")
    #
    # # Model 4: Damped trend + Multiplicative season with box-cox transformation
    # tes_model4 = ExponentialSmoothing(y_test, seasonal_periods=season, trend='add', seasonal='mul',
    #                                   damped_trend=True).fit()
    # y_pred_tes4 = tes_model4.predict(31924).rename("Model 4")
    #
    # fig = plt.figure(figsize=(60, 8))
    # tes_model1.fittedvalues.plot(color='blue')
    # y_pred_tes1.plot(color='blue', legend=True)
    # tes_model2.fittedvalues.plot(color='red')
    # y_pred_tes2.plot(color='red', legend=True)
    # tes_model3.fittedvalues.plot(color='green')
    # y_pred_tes3.plot(color='green', legend=True)
    # tes_model4.fittedvalues.plot(color='yellow')
    # y_pred_tes4.plot(color='yellow', legend=True)
    # fig.savefig('results/TES/final_output.jpg', bbox_inches='tight')
    # plt.title("Holt-Winters' Method/Triple Exponential Smoothing")
    # plt.show()
    #
    # print("Predicted values (Model 1): ", y_pred_tes1, "\n")
    # print("Predicted values (Model 2): ", y_pred_tes2, "\n")
    # print("Predicted values (Model 3): ", y_pred_tes3, "\n")
    # print("Predicted values (Model 4): ", y_pred_tes4, "\n")
    # mse_tes1 = mean_squared_error(y_test[31923:-1], y_pred_tes1)
    # mse_tes2 = mean_squared_error(y_test[31923:-1], y_pred_tes2)
    # mse_tes3 = mean_squared_error(y_test[31923:-1], y_pred_tes3)
    # mse_tes4 = mean_squared_error(y_test[31923:-1], y_pred_tes4)
    #
    # np.savetxt('results/TES/predicted_values_model1.txt', y_pred_tes1)
    # np.savetxt('results/TES/predicted_values_model2.txt', y_pred_tes2)
    # np.savetxt('results/TES/predicted_values_model3.txt', y_pred_tes3)
    # np.savetxt('results/TES/predicted_values_model4.txt', y_pred_tes4)
    # actual_test_result = y_test
    # np.savetxt('results/TES/test_values.txt', actual_test_result)
    #
    # return mse_tes1, mse_tes2, mse_tes3, mse_tes4

    return mse_tes, rmse_tes, r2_tes, y_pred_tes


alpha = 0.8
beta = 0.2
season = 24
y_test = pd.DataFrame(y_test)
y = np.reshape(np.array(y_test[327:-1]), (36,))
print("---------------------------------------------------------")

mse_ses, rmse_ses, r2_ses, y_ses = single_exponential_smoothing(alpha, y_test)
print("MSE for SES: ", mse_ses)
print('RMSE for SES:', rmse_ses)
print('R-squared for SES:', r2_ses)
print('MAPE for SES:', np.mean(np.abs((y - np.array(y_ses)) / y)) * 100,'\n')

print("---------------------------------------------------------")

mse_des, rmse_des, r2_des, y_des = double_exponential_smoothing(alpha, beta, y_test)
print("MSE for DES: ", mse_des)
print('RMSE for DES:', rmse_des)
print('R-squared for DES:', r2_des)
print('MAPE for DES:', np.mean(np.abs((y - np.array(y_des)) / y)) * 100,'\n')

print("---------------------------------------------------------")

mse_tes, rmse_tes, r2_tes, y_tes = triple_exponential_smoothing(season, y_test)
print("MSE for TES: ", mse_tes)
print('RMSE for TES:', rmse_tes)
print('R-squared for TES:', r2_tes)
print('MAPE for TES:', np.mean(np.abs((y - np.array(y_tes)) / y)) * 100,'\n')

print("---------------------------------------------------------")


# Plotting the results
fig = plt.figure(figsize=(60, 8))
plt.plot(y_ses, label='SES')
plt.plot(y_des, label='DES')
plt.plot(y_tes, label='TES')
plt.plot(y_test[327:-1], label='Actual Values')
plt.legend(loc='upper right')
plt.xlabel('Hour')
plt.ylabel('Electricity load')
plt.title("Predicted Values of various ES methods", fontsize=14)
plt.show()
fig.savefig('results/ESweather_final_output.jpg', bbox_inches='tight')