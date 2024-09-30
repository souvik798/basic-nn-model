# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value

## Neural Network Model


![image](https://github.com/user-attachments/assets/989bf911-b444-4e27-b16e-545f341a044a)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SOUVIK KUNDU
### Register Number:212221230105
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet=gc.open('newdata').sheet1
data=worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype(int)

x = df[['Input ']].values
y = df[['Output']].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
x_train=Scaler.transform(x_train)
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x_train,y_train,epochs=100)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(x_test)  
ai_brain.evaluate(X_test1, y_test)
X_n1 = [[3], [5]]

X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)



```
## Dataset Information

![def](https://github.com/user-attachments/assets/ceb858a7-b530-487e-ab11-72da935798ee)




## OUTPUT

### Training Loss Vs Iteration Plot

![def2](https://github.com/user-attachments/assets/b4055444-7807-4196-89f1-71d344a72640)



### Test Data Root Mean Squared Error

![def3](https://github.com/user-attachments/assets/93c0f269-eecc-4448-bb8c-e70445148e95)



### New Sample Data Prediction

![def4](https://github.com/user-attachments/assets/032290b4-fb56-49ca-a595-5b44c61a38a4)




## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
