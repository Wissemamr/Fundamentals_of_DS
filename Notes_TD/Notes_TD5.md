### Exercise 01 : 
Confusion Matrix Analysis
We have a binary classification model (e.g., logistic regression classifier), which has been trained
and tested on a small dataset of 15 emails drawn in next table, where Y and Y’ are the true labels
and the predicted labels respectively.


|  y  |  0    |  1   |1     |1     |0     |0     | 1    |1     |0     |0     |1     |1     | 0    |0     |1     |
|-----|------ |------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| y'  |   0   | 1    |0     |1     |0     |1     |    0 |  1   |0     |  0   | 1    |0     |0     |0     |1     |



1. Compute the confusion matrix using the provided Y and Y’.
2. Interpret the confusion matrix..
3. Calculate accuracy, precision, recall, and F1-score.
4. Visualize the confusion matrix using a heatmap.



## Exercise 02 : 
Regression
Given the next regression algorithm, and the next table.

|   X|   Y| Predicted Y |
|----|----|---------- |
|1   |2    | 1.8          |
|2   |4    |    2.6       |
|3   |5    |       3.4    |
|4   |4    |          4.2 |
|5   |6    |          5.0 |



1. Calculate the residuals.
2. Calculate MAE, RMSE of the model.
3. Calculate the likelihood of the model, given that the error follows a normal distribution.


## Solution : 


1. Calculate the residuals

|   $X$|   $Y$ | Predicted $\hat{Y}$| Residuals|
|----|---- |----------    |---------|
|1   |2    | 1.8          |    0.2  |
|2   |4    |    2.6       |    1.4  |
|3   |5    |       3.4    |    1.6  |
|4   |4    |          4.2 |   -0.2  |
|5   |6    |          5.0 |    1.0  | 

2. Calculate MAE, RMSE of the model.

$$ MSE = \frac{1}{n} \sum_{x=1}^{n} (Y_i - \hat{Y_i})$$


$$ MSE = \frac{1}{5} (0.2 + 1.4 + 1.6 -0.2 + 1.0)   $$
$$ MSE = 0,8 $$

<br>

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2} $$

$$ RMSE = \sqrt { \frac{1}{5} ( (0.2)^{2} + (1.4)^{2} + (1.6)^{2} +(-0.2)^{2} + (1.0)^{2} )} $$

$$RMSE = \sqrt { \frac{5.6}{5}}$$
$$RMSE = \sqrt {1.12}$$
$$RMSE \approx 1.0583 $$


3. Calculate the likelihood of the model, given that the error follows a normal distribution
The models  



Exercise 3: Model evaluation
1. Explain the relationship between model simplicity and underfitting. How does a model
being too simple contribute to underfitting?
2. How does the size of the dataset influence the likelihood of overfitting and underfitting?
3. Discuss two approaches to address underfitting and improve the performance of an underfit
model.
4. Explain the bias-variance trade-off and its role in overfitting and underfitting
5. Discuss the importance of model evaluation metrics in identifying and mitigating
overfitting and underfitting.