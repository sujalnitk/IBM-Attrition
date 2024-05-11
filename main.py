import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import warnings
warnings.filterwarnings('ignore') 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib


df = pd.read_csv("Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = df.drop(columns=['EmployeeCount' , 'Over18'] , axis = 1)
df['Attrition'] = df['Attrition'].map({"Yes":1,"No":0})
X = df.drop(columns=['Attrition'])
y = df['Attrition']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

Categorical_columns = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
transformer = ColumnTransformer(transformers=[
    ('tnf1',OneHotEncoder(sparse=False,drop='first'),Categorical_columns),
],remainder='passthrough')

X_train_final = transformer.fit_transform(X_train)
X_test_final = transformer.transform(X_test)

scaler = StandardScaler()
scaler.fit(X_train_final)

X_train_scaled = scaler.transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)


# Set class weights (increase weight for class 1)
model = SVC(kernel='linear', C=10, gamma='scale', class_weight={0: 1, 1: 3})
model.fit(X_train_scaled, y_train)

# Evaluate the new model
y_pred = model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_pred , y_test)

plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

# Loop through all cells and annotate them
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), ha='center', va='center', color='red')

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


print(f"accuracy : {accuracy_score(y_test,y_pred):.4f}")
print(f"recall : {recall_score(y_test,y_pred,average='binary'):.4f}")
print(classification_report(y_test, y_pred))