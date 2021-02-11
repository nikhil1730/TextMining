from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
import matplotlib.pyplot as plt
import warnings
import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

start = time.time()
training_tfidf, test_tfidf = load_svmlight_file('training_data_file.TFIDF')
featureSize = 20000
training_tfidf_chi2 = SelectKBest(chi2, k=featureSize).fit_transform(training_tfidf, test_tfidf)

clusterSize = []
kmeans_model_silhouette_score = []
kmeans_model_normalized_mutual_info_score = []
single_linkage_model_silhouette_score = []
single_linkage_model_normalized_mutual_info_score = []
for cluster in range(2,25):
    clusterSize.append(cluster)
    kmeans_model = KMeans(n_clusters=cluster).fit(training_tfidf_chi2)
    single_linkage_model = AgglomerativeClustering(n_clusters=cluster, linkage='ward').fit(training_tfidf_chi2.toarray())
    warnings.filterwarnings("ignore")

    kmeans_model_silhouette_score.append(metrics.silhouette_score(training_tfidf_chi2, kmeans_model.labels_, metric='euclidean'))
    kmeans_model_normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(test_tfidf, kmeans_model.labels_, average_method='arithmetic'))
    
    single_linkage_model_silhouette_score.append(metrics.silhouette_score(training_tfidf_chi2, single_linkage_model.labels_, metric='euclidean'))
    single_linkage_model_normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(test_tfidf, single_linkage_model.labels_, average_method='arithmetic'))
print()
print('ClusterSize Range for Feature Size ' + str(featureSize) +' is:' + str(clusterSize))  
print('Kmeans Model Silhouette Score:                    ' + str(kmeans_model_silhouette_score))
print('Kmeans Model Normalized Mutual Info Score:        ' + str(kmeans_model_normalized_mutual_info_score))
print('Single Linkage Model Silhouette Score:            ' + str(single_linkage_model_silhouette_score))
print('Single Linkage Model Normalized Mutual Info Score:' + str(single_linkage_model_normalized_mutual_info_score))
   
plt.xlabel('Cluster Size')
plt.ylabel('Method')

plt.subplot(2,1,1)
plt.title('Silhouette Measure')
plt.plot(clusterSize,kmeans_model_silhouette_score, label='KMeans')
plt.plot(clusterSize,single_linkage_model_silhouette_score, label='AgglomerativeClustering')
plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.subplot(2,1,2)
plt.title('Normalized Mutual Info')
plt.plot(clusterSize,kmeans_model_normalized_mutual_info_score, label='KMeans')
plt.plot(clusterSize,single_linkage_model_normalized_mutual_info_score, label='AgglomerativeClustering')
plt.tight_layout()
plt.legend(loc='best', shadow=True)

filename = 'clustering' + str(featureSize)+'.png'
plt.savefig(filename)
print('Saved the plot in the' + filename)

elapsed_time_fl = (time.time() - start) 
print(elapsed_time_fl)
    