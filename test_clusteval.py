
#%% Import libraries
from clusteval import clusteval
from df2onehot import df2onehot

# Load data from UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'

# Initialize clusteval
ce = clusteval()
# Import data from url
df = ce.import_example(url=url)

#%% Preprocessing
cols_as_float = ['ProductRelated', 'Administrative']
df[cols_as_float]=df[cols_as_float].astype(float)
dfhot = df2onehot(df, excl_background=['0.0', 'None', '?', 'False'], y_min=50, perc_min_num=0.8, remove_mutual_exclusive=True, verbose=4)['onehot']

#%%
# Initialize using the specific parameters
ce = clusteval(evaluate='silhouette',
               cluster='agglomerative',
               metric='hamming',
               linkage='complete',
               min_clust=2,
               verbose='info')

# Clustering and evaluation
results = ce.fit(dfhot)

# [clusteval] >INFO> Saving data in memory.
# [clusteval] >INFO> Fit with method=[agglomerative], metric=[hamming], linkage=[complete]
# [clusteval] >INFO> Evaluate using silhouette.
# [clusteval] >INFO: 100%|██████████| 23/23 [00:28<00:00,  1.23s/it]
# [clusteval] >INFO> Compute dendrogram threshold.
# [clusteval] >INFO> Optimal number clusters detected: [9].
# [clusteval] >INFO> Fin.
# %%
# Import library
from clusteval import clusteval
# Initialize clusteval with default parameters
ce = clusteval()

#%% Generate random data
from sklearn.datasets import make_blobs
X, labels_true = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)

#%% Fit best clusters
results = ce.fit(X)

#%% Make plot
ce.plot()

# silhouette plot
ce.plot_silhouette()

# Scatter plot
ce.scatter()

# Dendrogram
ce.dendrogram()
# %%
