import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Importing the California House Price Dataset
house_price_dataset = sklearn.datasets.fetch_california_housing()
house_price_dataframe = pd.DataFrame(house_price_dataset.data)

house_price_dataframe['price'] = house_price_dataset.target # add target (price) column to DataFrame

house_price_dataframe.shape  # (506,14) is total (rows, cols)
house_price_dataframe.isnull().sum()  # no missing values

house_price_dataframe.describe()  # gives statistical measures

correlation = house_price_dataframe.corr()  # correlation

plt.figure(figsize=(10,10))  # constructing
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# Splitting the data and target
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# Model Training - XGBoost Regressor
model = XGBRegressor()  # loading model
model.fit(X_train, Y_train)

# Evaluation
# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)

# R Squared Error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R Squared Error: ", score_1)
print("Mean Absolute Error: ", score_2)

# Prediction on test data

# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)
score_1 = metrics.r2_score(Y_test, test_data_prediction)
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R Squared Error: ", score_1)
print("Mean Absolute Error: ", score_2)

# Visualizing actual prices and predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Price")
plt.show()


