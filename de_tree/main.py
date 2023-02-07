import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("studdent.csv")

feature = ["total","days"]

x = data[feature]
y = data["gender"]

model = DecisionTreeClassifier()
dtree = model.fit(x,y)
tree.plot_tree(dtree,feature_names=feature)