from sklearn.cluster import KMeans
import matplotlib.pyplot as mb
x = [10090,565565,2000,4000,50090]
y = [4,5,4,3]

point = []

data = list(zip(x,y))
for x in range(1,5):
    kmeans = KMeans(n_clusters=x)
    kmeans.fit(data)
    point.append(kmeans.inertia_)
    
mb.plot(range(1,5),point,marker='o')