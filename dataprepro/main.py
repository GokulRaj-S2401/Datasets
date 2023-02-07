import pandas as pd
import numpy as np

data = pd.read_csv("datasets.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
x[:,0] = label.fit_transform(x[:,0])
y = label.fit_transform(y)
