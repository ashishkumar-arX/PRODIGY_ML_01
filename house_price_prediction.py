# implementing a linear regression model to predict the prices of 
# houses on their square footage and the number of 
# bedrooms and bathrooms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("task1\house_data.csv")
dataset = dataset[['bedrooms','bathrooms','sqft_living','sqft_lot','price']]

print(dataset.info())

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.005,random_state=22)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=0)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



x1 = [n for n in range(len(y_test))]  

# plt.scatter(x1, y_test, label='Accurate price', color='g')
plt.plot(y_test, marker = 'o', ms = 5, mec = 'g',mfc = 'g', label='Accurate price')  
  
# plt.scatter(x1, y_pred, label='Predicted price', color='r')  
plt.plot(y_pred, marker = 'o', ms = 5,mec = 'r',mfc = 'r', label='Predicted price') 

plt.xlabel('no. of houses -->')  
plt.ylabel('price -->')  
plt.title('Scatter Plot')  
plt.grid()
plt.legend()  
plt.show()  
