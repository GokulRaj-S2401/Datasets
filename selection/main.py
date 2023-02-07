import pandas as pd
from sklearn import datasets

datas = datasets.load_iris()
sw = datas.data
tar = datas.target


from sklearn.feature_selection import SelectKBest,chi2

test = SelectKBest(score_func=chi2,k=4)
fit = test.fit(sw,tar)
featureScore = fit.transform(sw)
print(featureScore)


