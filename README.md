# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```python

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harsha Vardhan
RegisterNumber:  212222240114
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![2 1](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/186fe800-8bcd-458e-92a3-585f85272a60)
![2 2](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/94161ef2-9a7e-411a-9f2e-e1ede287ade7)
![2 3](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/21625f34-6710-460d-b905-e63e592b8037)
![2 4](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/c8a9b905-3be9-4792-a7cd-4f887c034591)
![2 5](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/fc7ee781-5bbe-4f52-aa0f-017361fdda8b)

![2 6](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/0cdfce32-52b7-49cf-92ba-f9a5eb4a5a0a)

![2 7](https://github.com/Harshavardhan779/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707175/68bef5c8-8626-4313-8c4e-50a94e3bb686)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
