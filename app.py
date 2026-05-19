import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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

print("-----------Best Decision Tree---------------")
print("Best Parameters :",grid.best_params_)
print("Best Score:",grid.best_score_)
best_clf = grid.best_estimator_

# Assign the best model
DTC_model=best_clf

#Create Random Forest Classifier
RF_model_create = RandomForestClassifier(random_state=42)
#Parameter grid
param_dist={
    'n_estimators'      : [50,100,200,300],
    'max_depth'         : [3,5,10,20,None],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf'  : [1,2,4],
    'max_features'      : ['sqrt','log2']
}

random_search=RandomizedSearchCV(
    estimator=RF_model_create,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
random_search.fit(X_train,y_train)

print("-----------Best Random Forest---------------")
print(f"Best Parameters:{random_search.best_params_}")
print(f"Best Score:     {random_search.best_score_}")
RF_model=random_search.best_estimator_



# Model training
LR_model.fit(X_train,y_train)



#Prediction
LR_pred=LR_model.predict(X_test)
DTC_pred=DTC_model.predict(X_test)
RF_pred =RF_model.predict(X_test)

# Accuracy evaluation
LR_accuracy= round(accuracy_score(y_test,LR_pred)*100,2)
print(" LR Accuracy : ",LR_accuracy)
DTC_accuracy=round(accuracy_score(y_test,DTC_pred)*100,2)
print(" DTC Accuracy : ",DTC_accuracy)
RF_accuracy=round(accuracy_score(y_test,RF_pred)*100,2)
print(f"RF Accuracy : {RF_accuracy}")

#Model ranking
results = pd.DataFrame({
    'Model': ['LR_model','DTC_model','RF_model'],
    'Accuracy': ['LR_accuracy','DTC_accuracy','RF_accuracy']
})

print(results.sort_values(by='Accuracy', ascending=False))



LR_cm = confusion_matrix(y_test,LR_pred)
DTC_cm=confusion_matrix(y_test,DTC_pred)
RF_cm=confusion_matrix(y_test,RF_pred)
print("Logistic Regression Confusion Matrix")
print(LR_cm)
print("Decision Tree Classifier Confusion Matrix")
print(DTC_cm)
print("Random Forest Classifier")
print(RF_cm)


fig,axes=plt.subplots(1,3,figsize=(15,4))
#Logistic Regression Confusion Matrix
LR_disp=ConfusionMatrixDisplay(confusion_matrix=LR_cm,display_labels=['No','Yes'])
LR_disp.plot(cmap='Blues',ax=axes[0])
axes[0].set_title('Logistic Regression')

#Decision Tree Classifier Confusion Matrix
DTC_disp=ConfusionMatrixDisplay(confusion_matrix=DTC_cm,display_labels=['No','Yes'])
DTC_disp.plot(cmap='Blues',ax=axes[1])
axes[1].set_title('Decision Tree')

#Random Forest Classifier Confusion Matrix
RF_disp=ConfusionMatrixDisplay(confusion_matrix=RF_cm,display_labels=['No','Yes'])
RF_disp.plot(cmap='Blues',ax=axes[2])
axes[2].set_title('Random Forest')

plt.suptitle('Confusion Matrices Comparison',fontsize=16,fontweight='bold')
plt.tight_layout()



#Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': RF_model.feature_importances_
})

print(feature_importance.sort_values(by='Importance', ascending=False))

#Accuracy Comparison

models = ['Logistic Regression','Decision Tree','Random Forest']
accuracies=[LR_accuracy,DTC_accuracy,RF_accuracy]
colors = ['#378ADD', '#1D9E75', '#7F77DD']

plt.figure(figsize=(8,5))
bars = plt.bar(models,accuracies,color=colors,edgecolor='white',width=0.5)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() /2,
        bar.get_height() + 0.5 ,
        f'{bar.get_height():.2f}%',
        ha='center',fontsize=11
    )

plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(60,105)
plt.tight_layout()
plt.show()

#Random Forest and Logistics Regression achieved the best performance.

