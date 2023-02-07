import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas_profiling as pp 
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score,roc_curve,roc_auc_score

df = pd.read_csv('./diabetes2.csv')
df_temp = df.copy()

sns.catplot(x="Outcome", kind="count", data=df_temp, palette="Set2")
plt.show()

x = df_temp.drop(['Outcome'], axis = 1)
y = df_temp.loc[:,"Outcome"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 123)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

lr = LogisticRegression(solver='liblinear', max_iter = 1000)

lr.fit(x_train, y_train)

lr.fit(x_train, y_train)

x_pred = lr.predict(x_train)

data = [[5, 150, 33.7, 50, 150, 74, 0.5, 53]]

# Create the pandas DataFrame 
df_test = pd.DataFrame(data, columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

# Predict on new data
res = lr.predict(df_test)
print(res)