import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')



# Data cleaning
df['TotalCharges']= pd.to_numeric(df['TotalCharges'],errors='coerce')
df=df.dropna()


# Feature selection
X = df[['tenure','MonthlyCharges','TotalCharges']]
y = df['Churn']

y = y.map({'Yes':1,'No':0})

# Train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Create model
model=LogisticRegression()

# Model training
model.fit(X_train,y_train)

#Prediction
predictions=model.predict(X_test)

# Accuracy evaluation
accuracy= accuracy_score(y_test,predictions)
print("Accuracy : ",accuracy)