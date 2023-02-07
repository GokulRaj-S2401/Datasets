import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn import datasets
data = datasets.load_iris()
x = data.data
y = data.target

model = GaussianMixture(n_components=3,random_state=3000)
model.fit(x)
uu = model.predict(x)

cm = confusion_matrix(uu, y)
print(cm)
score = accuracy_score(uu, y)
print(score)
