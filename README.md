# California House Price Prediction with XGBoost

This project utilizes the XGBoost algorithm to predict house prices in California based on various features. It demonstrates how to preprocess the data, train the model, evaluate its performance, and visualize the results using Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and XGBoost.

## Project Overview

The goal of this project is to build a regression model that accurately predicts house prices using the California House Price Dataset. The dataset contains various features related to houses in California, and the target variable is the corresponding house price. The steps involved in this project are as follows:

1. Data Preprocessing: The dataset is loaded, inspected for missing values, and prepared for model training.

2. Exploratory Data Analysis: Correlation between features is visualized using a heatmap to gain insights into relationships among variables.

3. Model Training: An XGBoost Regressor is used to train the predictive model.

4. Model Evaluation: The model's accuracy is evaluated on both the training and test datasets using R Squared Error and Mean Absolute Error metrics.

5. Visualizing Results: A scatter plot is created to visualize the actual prices against the predicted prices.

## How to Use

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your_username/California-House-Price-Prediction.git
```

2. Install the required libraries (if not already installed) using:

```bash
pip install numpy pandas matplotlib seaborn xgboost scikit-learn
```

3. Run the Python script:

```bash
python house_price_prediction.py
```

4. The script will output the R Squared Error and Mean Absolute Error for both the training and test datasets. It will also display a scatter plot showing the relationship between actual prices and predicted prices.

## License

This project is licensed under the [MIT License](LICENSE).

