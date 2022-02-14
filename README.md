# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step 1:
Import Pandas module
### Step 2:
Import linear_module from sklearn
### Step 3:
Declare data frame df to read the csv file
### Step 4:
Declare variable X equal to df with two arguments Weight and Volume
### Step 5:
Declare variable y equal to df with an argument CO2
### Step 6:
Declare a variable regr equal to linear_model.LinearRegression()
### Step 7:
declare regr.fit(X,y)
### Step 8:
Print regr.coeff_
### Step 9:
Print regr.intercept_
### Step 10:
Declare variable predictedCO2 equal to regr.predict with two arguments 33300, 1300
### Step 11:
Print variable predictedCO2

## Program:
~~~
"Cars.Csv"
V\import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df=pd.read_csv("cars.csv")

X=df[['Weight','Volume']]
y=df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X,y)

#Coefficients and intercepts of Model

print('Coefficients: ', regr.coef_)
print('Intercept:',regr.intercept_)

#predict the c02 emission of a car where the weight is 300kg,and the volume is 1300cm3

predictedCO2= regr.predict([[330,130]])

print('Predicted CO2 for the corresponding weight and volume',predictedCO2)

"Clustering.Csv"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
X1 = pd.read_csv("clustering.csv")
print (X1. head(2))
X2 = X1.loc[:, ['ApplicantIncome', 'LoanAmount' ]]
print(X2. head(2))
X = X2.values
sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show( )
kmean=KMeans(n_clusters=4)
kmean. fit(X)
print('Cluster Centers: ',kmean.cluster_centers_)
print('Labels: ',kmean.labels_)
# predict the class for ApplicantIncome 9060 and Loanamount 120
predicted_class = kmean.predict([[9000, 120]])
print("The cluster group for Applicant Income 9060 and Loanamount 120 is",predicted_class)
~~~

## Output:
![GitHub Logo](Ex10(1).png)
![GitHub Logo](Ex10(2).png)
## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
