import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def dbscan_lib(epsilon, min_sample, X):
  # #############################################################################
  # Compute DBSCAN
  cluster_model = DBSCAN(eps=epsilon, min_samples=min_sample).fit(X)
  core_samples_mask = np.zeros_like(cluster_model.labels_, dtype=bool)
  core_samples_mask[cluster_model.core_sample_indices_] = True
  labels = cluster_model.labels_

  clusters = cluster_model.labels_.tolist()
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

  return clusters, n_clusters

  
