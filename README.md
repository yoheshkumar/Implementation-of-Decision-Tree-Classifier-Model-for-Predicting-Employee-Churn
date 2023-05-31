# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import standard libraries in python for finding Decision tree classsifier model for predicting employee churn.
I2. nitialize and print the Data.head(),data.info(),data.isnull().sum()
3. Visualize data value count.
4. Import sklearn from LabelEncoder.
5. Split data into training and testing.
6. Calculate the accuracy, data prediction by importing the required modules from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: yohesh kumar R.M
RegisterNumber: 212222240118 
*/
```
```

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```


## Output:


### Initial data set:
![ids](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/8e1330e0-87f5-4c07-9103-f37661e36594)
### Data info:
![di](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/755b9f27-9769-4fcf-8fe9-72e1cea9624a)
### Optimization of null values:
![onv](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/cb42228b-1bd6-43ba-8230-2cf059215b7f)
### Assignment of x and y values:
![yx](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/d4733110-5cfd-481f-97d7-47f9e0df9adc)
### data.head() for salary:
![ukn](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/99a3f2ac-99d0-4669-a7f1-0c4eaa14fffc)

Converting string literals to numerical values using label encoder:
![num lablel encoder](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/fee4b82a-300e-4268-aba2-86b62db247f9)
### accuracy
![acc](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/e6877d1e-99aa-4e8a-b098-2c93dde45783)
### prediction
![pred](https://github.com/yoheshkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393568/27c3c439-c25a-4159-945e-81bc1a9f7054)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
