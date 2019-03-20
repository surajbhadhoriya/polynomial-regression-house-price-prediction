#####################POLYNOMIAL REGRESSION ################################
#import all require library
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#load data 
data1=pd.read_csv('C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/kc_house_data.csv')
data1.info()
x=data1['sqft_living']
y=data1['price']
x1=sorted(x,key=None,reverse=False)
y1=sorted(y,key=None,reverse=False)
x1=np.array(x1,dtype=np.int64)
y1=np.array(y1,dtype=np.int64)
#plot data
plt.scatter(x1,y1,color='green')
#apply polynomial regression with degree 5
from sklearn.preprocessing import PolynomialFeatures
pol_reg= PolynomialFeatures(degree=5)
x1=x1.reshape(-1,1)
x_pol=pol_reg.fit_transform(x1)
#split data set for training and testing
X_train, X_test, y_train, y_test=train_test_split(x_pol, y1, test_size=0.2, random_state=5)
#apply model
model=LinearRegression()
model.fit(X_train, y_train)
b1=model.intercept_
m1=model.coef_
print("intercept",b1)
print("slope",m1)
ac1=model.score(X_test,y_test)
print("accuracy",ac1)
y_pred1=model.predict(X_test)
print("prediction",y_pred1)





plt.scatter(x1,y1,color='red')
plt.plot(x1,model2.predict(pol_reg.fit_transform(x1)),color='blue')
plt.tittle("Truth or bbluff (linear regression)")
plt.xlabel("squarfit_living")
plt.ylabel("price")
plt.show()

