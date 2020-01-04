
#import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset

dataset = pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,1].values

#splitting the dataset into the training set and test set

from sklearn.cross_validation import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0 )

 #fitting simple linear regression
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the Test set results

y_pred = regressor.predict(x_test)

x_pred = regressor.predict(x_train)
#visualising the training set results


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, x_pred ,color='blue')
plt.title('Salary vs exp')
plt.xlabel('years')
plt.ylabel('salary')
plt.show()

  




