import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from scipy.spatial.distance import  cdist        #introducing the function of distance

#加载数据
x,y=make_blobs(n_samples=100,centers=6,random_state=1234,cluster_std=0.6)

##draw scatered picture
#plt.figure(figsize=(6,6))
#plt.scatter(x[:,0],x[:,1],c=y)
#plt.show()

##算法实现
class Kmeans(object):
    #initial,parameter:n_cluster(K),最大迭代次数max_iter,初始质心centroids
    def __init__(self,n_cluster=6,max_iter=300,centroids=[]):
        self.n_cluster=n_cluster
        self.max_iter=max_iter
        self.centroids=np.array(centroids,dtype=float)
    #训练模型方法
    def fit(self,data):
        #随机选取质心
        if(self.centroids.shape==(0,)):
            #从data中随机选择n_cluster个质心，选择的范围0~data矩阵的行数
            self.centroids=data[np.random.randint(0,data.shape[0],self.n_cluster),:]

            #begin training
        for i in range(self.max_iter):
             #1.计算距离矩阵,得到100*6的矩阵
            distances=cdist(data,self.centroids)
             #2.对距离排序，设当前点为最近点质心的cluster
            c_ind=np.argmin(distances,axis=1)   #c_ind是一个100*1的标签矩阵，axis=1表示保留一列

            #3.计算均值，更新质心点坐标
            for i in range(self.n_cluster):
                #遍历所有类，如果没有一个样本分类标记为这个类，那就没有必要更新这个类
                if i in c_ind:
                    #更新第i个类的质心
                    self.centroids[i]=np.mean(data[c_ind==i],axis=0)   #布尔索引

    def predict(self,samples):

        distances= cdist(samples,self.centroids)
        c_ind=np.argmin(distances,axis=1)

        return c_ind

#绘制子图函数
def plotKMeans(x,y,centroid,subplot,title):
        plt.subplot(subplot)
        plt.scatter(x[:,0],x[:,1],c='r')

        plt.scatter(centroid[:,0],centroid[:,1],c=np.array(range(6)),s=100)
        plt.title(title)
        plt.show()

kmeans=Kmeans(max_iter=300,centroids=np.array([[2,1],[2,2],[2,3],[2,4],[2,5],[2,6]]))
plt.figure(figsize=(6,6))

#121表示画出一行两列的第1个子图
plotKMeans(x,y,kmeans.centroids,121,'initial state')
kmeans.fit(x)
plotKMeans(x,y,kmeans.centroids,122,'final state')