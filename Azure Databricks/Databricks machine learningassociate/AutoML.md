Databricks AutoML helps you automatically apply machine learning to a dataset

You provide the dataset and identify the prediction target, while AutoML prepares the dataset for model training

AutoML then performs and records a set of trials that creates, tunes and revaluates multiple models

After model evaluation, AutoML displays the results and provides a Python notebook with the source code for  each trial run so you can review, reproduce and modify the code

AutoML also calculates summary statistics on your  dataset and saves this information in a notebook that you can review later

## How does it work ?

- Prepare the dataset for model training : Data imbalance detection
- Iterates to train and tune multiple models: Hyper-parameter tuning.
- Evaluates models based on algorithms
- Display the results and provides a Python Notebook : Notebook for each trial launched .

AutoML algorithms : 

| Classification      | Regression                                           | Forecasting Models |
| ------------------- | ---------------------------------------------------- | ------------------ |
| Decision trees      | Decision trees                                       | Prophet            |
| Random forest       | Random forest                                        | Auto-ARIMA         |
| logistic regression | logistic regression with stochastic gradient descent |                    |
| XGBoost             | XGBoost                                              |                    |
| LightGBM            | LightGBM                                             |                    |
