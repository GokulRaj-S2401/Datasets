import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mb

x = [5.5,6,4.8,5,5.8,6.8,6.8,4,4.7,5.1,4]
y = [21,21,20,24,25,35,21,23,45,43,56]
clas = [1,1,0,1,1,1,0,0,0,0,0]

data = list(zip(x,y))
model = KNeighborsClassifier(n_neighbors=5)
newX_value = 3.8
newY_value = 59

model.fit(data,clas)
newPoint = [(newX_value,newY_value)]
predict = model.predict(newPoint)
print(predict[0])
