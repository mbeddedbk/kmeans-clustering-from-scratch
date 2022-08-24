
import random

random.seed(10)

def distance(x1, x2):

    n = range(len(x1))
    eucDis = 0

    for i in n:
        eucDis+= (x1[i] - x2[i])**2
    
    eucDis = (eucDis)**0.5

    return eucDis


class KMeansClusterClassifier():
    def __init__(self, n_cluster = 3):
        
        self.n_cluster = n_cluster
        self.max_iters = 50000

        #Cluster sayisi kadar bos list olusturulacak
        #List of sample indices for each cluster
        self.clusters = [[] for _ in range (self.n_cluster)]
        #Her clusterin mean degeri burada tutulacak
        self.centroids = []
    
    def predict(self, x_test):  # 2 1 0

        clustred_samples = [[] for _ in range(self.n_cluster)]

        for i in range(len(x_test)): 
            tempa = [0] * 3
            for elem_indx, elem in enumerate(self.centroids):
                zer_cluster = (x_test[i][0] - elem[0])**2 + (x_test[i][1] - elem[1])**2 + (x_test[i][2] - elem[2])**2 + (x_test[i][3] - elem[3])**2
                tempa[elem_indx] = zer_cluster
            x = tempa.index(min(tempa))
            clustred_samples[x].append(x_test[i])
        
        return clustred_samples
    
    def fit(self, X):
        self.X = X
        self.n_data = len(X)
        self.n_features = len(X[0]) #Burada etiketleri gondermedigine dikkat et

        # centroidlerin yapilandirilmasi
        rnd_indxs = random.sample(range(1, self.n_data), self.n_cluster)     
        self.centroids = [self.X[idx] for idx in rnd_indxs]       

        #optimization
        for _ in range(self.max_iters):

            #obeklerin guncellenmesi
            self.clusters = self._create_clusters(self.centroids)

            #centeroidlerin guncellenmesi
            old_centeroids = self.centroids
            self.centroids = self._get_centeroids(self.clusters)

            #centeroidler yakinsadi mi
            if self._is_converged(old_centeroids, self.centroids):
                break

        fitted_clusters = [[] for _ in range (self.n_cluster)]
        i = 0
        for fitted_cluster_idx, fitted_cluster in enumerate(self.clusters):
            for i in range(len(fitted_cluster)):
                fitted_clusters[fitted_cluster_idx].append(X[fitted_cluster[i]])
        return fitted_clusters

        #cluster etiketlerinin geri donulmesi
        #return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = []
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels.append(cluster_idx)
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range (self.n_cluster)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centeroids):
        distances = [distance(sample, point) for point in centeroids]
        closest_idx = distances.index(min(distances))
        return closest_idx

    def _get_centeroids(self, clusters):
        
        centroids = []
        temp_list = [0] * self.n_features
        for _ in range(self.n_cluster):
            centroids.append(temp_list)

        for cluster_idx, cluster in enumerate(clusters):
            temp_cluster_mean = []
            for i in range(len(cluster)):
                temp_cluster_mean.append(self.X[cluster[i]])
            avg = [float(sum(col))/len(col) for col in zip(*temp_cluster_mean)]
            centroids[cluster_idx] = avg   
        return centroids

    def _is_converged(self, cent_old, cents):
        distances = [distance(cent_old[i], cents[i]) for i in range(self.n_cluster)]
        return sum(distances) == 0

