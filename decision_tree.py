# polynomial regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

#feature scaling
#from sklearn.preprocessing import StandardScaler

#sc_x = StandardScaler()
#sc_y = StandardScaler()
#x = sc_x.fit_transform(x)
#y = sc_y.fit_transform(y)

#fitting linear regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)



y_pred = regressor.predict(6.5)

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth or bluff')
plt.xlabel('position level')
ptl.ylabel('salary')
plt.show()


