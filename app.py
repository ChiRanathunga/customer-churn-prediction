import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')



# Data cleaning
df['TotalCharges']= pd.to_numeric(df['TotalCharges'],errors='coerce')
df=pd.get_dummies(df,columns=['Contract','InternetService'],dtype=int)
df=df.dropna()


# Feature selection
X = df[['tenure','MonthlyCharges','TotalCharges','Contract_Month-to-month','Contract_One year','Contract_Two year','InternetService_DSL','InternetService_No','InternetService_Fiber optic','SeniorCitizen']]
y = df['Churn']

y = y.map({'Yes':1,'No':0})

# Train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Create Logistic Regression model  
LR_model=LogisticRegression()

# Hyperparameter values to test for Decision Tree
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}

# Perform Grid Search to find the best Decision Tree parameters
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, 
    cv=5,
    scoring='accuracy'
)
grid.fit(X_train, y_train)

print("Best Parameters:",grid.best_params_)
print("Best Accuracy:",grid.best_score_)
best_clf = grid.best_estimator_

# Assign the best model
DTC_model=best_clf



# Model training
LR_model.fit(X_train,y_train)
DTC_model.fit(X_train,y_train)

#Prediction
LR_pred=LR_model.predict(X_test)
DTC_pred=DTC_model.predict(X_test)

# Accuracy evaluation
LR_accuracy= accuracy_score(y_test,LR_pred)
print(" LR Accuracy : ",LR_accuracy)
DTC_accuracy=accuracy_score(y_test,DTC_pred)
print(" DTC Accuracy : ",DTC_accuracy)



LR_cm = confusion_matrix(y_test,LR_pred)
DTC_cm=confusion_matrix(y_test,DTC_pred)
print("Logistic Regression Confusion Matrix")
print(LR_cm)
print("Decision Tree Classifier Confusion Matrix")
print(DTC_cm)



LR_disp=ConfusionMatrixDisplay(confusion_matrix=LR_cm,display_labels=['No','Yes'])
LR_disp.plot(cmap='Blues')
DTC_disp=ConfusionMatrixDisplay(confusion_matrix=DTC_cm,display_labels=['No','Yes'])
DTC_disp.plot(cmap='Blues')
plt.show()
