import csv
from dbscan_lib import *
from pprint import pprint
from sklearn import cluster as skcluster

import numpy as np
import pandas as pd
import random

def readFile(dataname):
  filedata = []
  with open(str(dataname), 'r') as csvfile:
    lines = csv.reader(csvfile)
    
    for idx,row in enumerate(lines):
      if idx not in (0,18047):
        filedata.append([float(row[4]), float(row[5])])
  return filedata

def automatic_clustering(num_cluster, data):
  #data = np.array(data)
  result = skcluster.AgglomerativeClustering(n_clusters=num_cluster,linkage='complete').fit_predict(data)
  print(result) 
  
if __name__ == "__main__":
  strdata = './Dataset/raw_isc_2018_2015.csv'
  
  # reading CSV file
  print("reading CSV file")
  data = pd.read_csv(strdata)

  print("converting to array list")
  data_cluster = data[["LAT","LON"]].values
  print(data_cluster[:10])
  
  print("clustering data")
  num_cluster = 4
  automatic_clustering(num_cluster, data_cluster)
  
  
  
  #result, n_cluster = dbscan_lib(0.3,10,data_cluster)
  #data['cluster'] = result

  #print(data)
  #map_plot(data, n_cluster)