import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Pre-processing of the data
df_raw = pd.read_csv('assets/hourly_loaddata.csv', header=None, skiprows=1)  # loading raw data from the CSV
df_raw_array = df_raw.values  # numpy array
y = np.array(df_raw[2])
temp = []
for i in range(y.size):
    temp.append([df_raw[0][i],df_raw[1][i]])
x = np.array(temp)


print(x)
print(y)
model = LinearRegression().fit(x,y)
y_pred = model.predict(x)
print('Predicted values:', y_pred)
print('Intercept:', model.intercept_)
print('Slope:', model.coef_)


print('The MSE value is:',
      mean_squared_error(y_pred, y, squared=True))
print('The RMSE value is:',
      mean_squared_error(y_pred, y, squared=False))
print('The R-squared value is:', r2_score(y_pred, y))
print('The MAPE value is:', np.mean(np.abs((y - y_pred) / y)) *100,'\n')

# Plotting the results
fig = plt.figure()
plt.plot((y_pred))
plt.plot((y))
plt.title("Baseline Model")
plt.xlabel('Hour')
plt.ylabel('Electricity load')
plt.legend(('Predicted', 'Actual'), fontsize='15')
plt.show()
fig.savefig('results/baseline_model/final_output.jpg', bbox_inches='tight')


# Storing the result in a file: 'load_forecasting_result.txt'
predicted_test_result = y_pred
np.savetxt('results/baseline_model/predicted_values.txt', predicted_test_result)
actual_test_result = y
np.savetxt('results/baseline_model/test_values.txt', actual_test_result)


