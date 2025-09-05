# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DEEPIKA R 
RegisterNumber: 212224230054

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
        return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
Data Information :

<img width="633" height="132" alt="Screenshot 2025-09-05 193013" src="https://github.com/user-attachments/assets/67ab8b27-01fc-4a69-8afb-92e2954389eb" />

Value of X:

<img width="690" height="790" alt="Screenshot 2025-09-05 193036" src="https://github.com/user-attachments/assets/26d2fb4e-d09c-4d7a-b670-6e17409e2313" />

Value of X1_Scaled:

<img width="722" height="788" alt="Screenshot 2025-09-05 193054" src="https://github.com/user-attachments/assets/b43a3df6-75bf-495b-bebb-d5c8521482e7" />

Predicted Value :

<img width="601" height="52" alt="Screenshot 2025-09-05 193114" src="https://github.com/user-attachments/assets/4a8985fb-efae-408d-a4cc-87c7cdbec3a5" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
