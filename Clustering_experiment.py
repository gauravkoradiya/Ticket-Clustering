import pickle
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, SpectralClustering
from nltk.cluster import KMeansClusterer,EMClusterer,GAAClusterer
from sklearn.metrics import davies_bouldin_score,silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import nltk
from nltk.corpus import stopwords

def k_means_experiment(sample,distance:['euclidean_distance','cosine_distance']='cosine_distance', min_k=2, max_k=50):

    score=[]
    silhouette=[]
    k_assigment_cluster=[]
    if distance == 'euclidean_distance':

         for k in range(min_k,max_k):
             kclusterer = KMeansClusterer(num_means=k, distance=nltk.cluster.util.euclidean_distance,avoid_empty_clusters=True,repeats=1)
             assigned_clusters = kclusterer.cluster(sample,assign_clusters=True)
             # kmeans = KMeans(n_clusters=k,random_state=5,n_jobs=-1,n_init=20,max_iter=500).fit(tickets_vec)
             # labels = np.array(kmeans.labels_)
             silhouette.append(silhouette_score(X=sample, labels=np.array(assigned_clusters)))
             score.append(davies_bouldin_score(sample, assigned_clusters))
             k_assigment_cluster.append(assigned_clusters)

    else:

        for k in range(min_k,max_k):
            #kmeans = KMeans(n_clusters=k,random_state=5,n_jobs=-1,n_init=20,max_iter=500).fit(tickets_vec)
            #assigned_clusters  = kmeans.labels_
            kclusterer = KMeansClusterer(num_means=k, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True,repeats=1)
            assigned_clusters = kclusterer.cluster(sample,assign_clusters=True)
            silhouette.append(silhouette_score(X=sample, labels=np.array(assigned_clusters)))
            score.append(davies_bouldin_score(sample, assigned_clusters))
            k_assigment_cluster.append(assigned_clusters)

    plt.plot(np.arange(min_k,max_k),np.array(score), label='davis bouldin score')
    plt.plot(np.arange(min_k,max_k),np.array(silhouette),label='silhouette score')
    plt.xlabel('number of cluster')
    plt.ylabel('DAVIES BOULDIN SCCORE')
    plt.title('K-means Cluster Scoring')
    plt.legend()
    plt.show()

    return kclusterer,k_assigment_cluster


def EMclusterer_experiment(samples,means):

    # silhouette=[]
    # devis_boldin=[]
    #for i in range(len(means)):
    emclusterer = EMClusterer(means)
    assigned_cluster = emclusterer.cluster(samples,assign_clusters=True)
    print(metrics.silhouette_score(X=samples,labels=assigned_cluster, metric='euclidean'))
    print(davies_bouldin_score(samples, assigned_cluster))
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # plt.plot(np.arange(2,len(means)),silhouette, label='silhouette')
    # plt.plot(np.arange(2,len(means)), devis_boldin, label='devis_bouldin')
    # plt.xlabel('number of cluster')
    # plt.ylabel('Score')
    # plt.title('EMCluster')
    # plt.legend()
    # plt.show()
    return emclusterer

def Gaaclusterer_experiment(samples,k_cluster):
    silhouette = []
    devis_boldin = []

    for i in range(2,k_cluster):
        gaaclusterer= GAAClusterer(num_clusters=i)
        assigned_cluster = gaaclusterer.cluster(samples,True)
        silhouette.append(metrics.silhouette_score(X=samples, labels=np.array(assigned_cluster)))
        devis_boldin.append(davies_bouldin_score(samples, assigned_cluster))

    plt.plot(np.arange(2,k_cluster), silhouette,c='r', label='silhouette')
    plt.plot(np.arange(2,k_cluster), devis_boldin,c='g' ,label='devis_bouldin')
    plt.xlabel('number of cluster')
    plt.ylabel('Score')
    plt.title('GAACluster')
    plt.legend()
    plt.show()
    return assigned_cluster

def agglomerative_experiment(samples,affinity:['euclidean', 'l1', 'l2','manhattan','cosine']='euclidean',min_k=2,max_k=50):

    silhouette = []
    devis_boldin = []

    for i in range(min_k,max_k):
        agglomeratrive_clusterer = AgglomerativeClustering(n_clusters=i,affinity=affinity,linkage='complete').fit(samples)
        assigned_cluster = agglomeratrive_clusterer.labels_
        silhouette.append(metrics.silhouette_score(X=samples, labels=np.array(assigned_cluster)))
        devis_boldin.append(davies_bouldin_score(samples, assigned_cluster))
    print(agglomeratrive_clusterer)
    plt.plot(np.arange(2,max_k), silhouette,c='r', label='silhouette')
    plt.plot(np.arange(2,max_k), devis_boldin,c='g' ,label='devis_bouldin')
    plt.xlabel('number of cluster')
    plt.ylabel('Score')
    plt.title('Agglomaretive Clustering')
    plt.legend()
    plt.show()
    return assigned_cluster

def birch_experiement(samples,n_cluster):
    silhouette = []
    devis_boldin = []

    for i in range(2,n_cluster):
        brc = Birch(threshold=0.5,branching_factor=10, n_clusters=i, compute_labels=True)
        assigned_cluster = brc.fit_predict(samples)
        silhouette.append(metrics.silhouette_score(X=samples, labels=np.array(assigned_cluster)))
        devis_boldin.append(davies_bouldin_score(samples, assigned_cluster))
    print(brc)
    plt.plot(np.arange(2,n_cluster), silhouette,c='r', label='silhouette')
    plt.plot(np.arange(2,n_cluster), devis_boldin,c='g' ,label='devis_bouldin')
    plt.xlabel('number of cluster')
    plt.ylabel('Score')
    plt.title('Birch Clustering')
    plt.legend()
    plt.show()
    return assigned_cluster

