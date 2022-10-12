import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv('./gld_price_data.csv')

print(gold_data.head())

print(gold_data.tail())

print(gold_data.shape)

print(gold_data.info())

print(gold_data.isnull().sum())

print(gold_data.describe())

correlation = gold_data.corr()

# constructing a heat map to understand the correlation

plt.figure(figsize= (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


print(correlation['GLD'])

# check the distribution of the gold price

sns.displot(gold_data['GLD'],color='green')

# splitting the features and the target

X = gold_data.drop(['Date','GLD'], axis=1)
Y = gold_data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

# Model training:
# Random forest regreasor model

regressor = RandomForestRegressor(n_estimators=100)

# training the model 

regressor.fit(X_train,Y_train)

# model evaluation 

# prediction on test data 

test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)

error_score = metrics.r2_score(Y_test, test_data_prediction)

print("R square error", error_score)

# compare the actual values and predicted values in a plot 

Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title("actual Price vs Predicted Price")
plt.xlabel("Number of Values")
plt.ylabel("GLD Price")
plt.legend()
plt.show()

