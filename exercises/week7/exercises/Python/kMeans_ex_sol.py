import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import operator
import matplotlib.pyplot as plt


cls = pd.read_csv("Data/ziplabel.csv", header=None)
X = np.asarray(pd.read_csv("Data/zipdata.csv", header=None))

[N, p] = X.shape

minX = list(np.min(X, axis=0)) # data range min
maxX = list(np.max(X, axis=0)) # data range max

clustersNr = 10
Nsim = 5
Wu = np.zeros((clustersNr, Nsim))
for nrClusters in range(clustersNr + 1): # Want actual number included
    kmeans = KMeans(n_clusters=nrClusters + 1, n_init = 5, n_jobs = -1).fit(X)
    C = kmeans.cluster_centers_ # the cluster centers in the p dimensions
    labelCluster = kmeans.labels_ # the labelling for each point

    # Compute within-class dissimilarity given X (the data), C (the cluster centers)
    # and gr (the predicted cluster numbers)
    W = []
    for cluster in range(1, nrClusters + 1):
        Ik = np.where(labelCluster == cluster - 1)
        dk = np.sum(X[Ik, :] - np.multiply(np.ones((np.size(Ik), 1)), C[cluster - 1, :])**2, axis = 1)
        Dk = np.sum(dk)
        W.append(Dk)

    # gap-statistic
    # Nsim simulations of data uniformly distributed over [X]
    for j in range(5):
        # simulate uniformly distributed data
        Xu = np.ones((N,1))*minX + np.random.rand(N,p)*(np.ones((N,1))*maxX-np.ones((N,1))*minX)
        # perform K-means
        kmeansU = KMeans(n_clusters=nrClusters + 1, n_init = 5, n_jobs = -1).fit(Xu)
        Cu = kmeansU.cluster_centers_
        labelClusterU = kmeansU.labels_

        # Compute within-class dissmiliarity for the simulated data given Xu (the simulated data),
        # Cu (the cluster centers for the simulated data), and gru (the predicted cluster numbers)
        # for the simulated data).

        for cluster in range(1, nrClusters+1):
            Ik = np.where(labelClusterU == cluster - 1)
            dku = np.sum(X[Ik, :] - np.multiply(np.ones((np.size(Ik), 1)), Cu[cluster - 1, :])**2, axis = 1)
            Dku = np.sum(dku)
            Wu[nrClusters - 1, j] = Wu[nrClusters - 1, j] + Dku

# compute expectation of simulated within-class dissimilarities, and the
# standard errors for the error bars
Elog_Wu = np.mean(np.log(np.abs(Wu)), axis = 1)
sk = np.std(np.log(np.abs(Wu)), axis=1)*np.sqrt(1+1/Nsim) # standard error sk' in (14.39)
x_range = np.array(range(nrClusters)) + 1

# Plot the log within class scatters
plt.figure()
plt.title("Within-class dissimilarity")
plt.plot(x_range, np.log(np.abs(W))[::-1], label='observed')
plt.plot(x_range, Elog_Wu[::-1], label='expected for simulation')
plt.legend(loc='upper left')
plt.xlabel("Number of clusters - k")
plt.xlabel("Number of clusters - k")
plt.ylabel("log(W)")
plt.show()

# plot the Gap curve
plt.figure()
plt.title('Gap curve')
Gk = map(operator.sub, Elog_Wu.T, np.log(np.abs(W)))
plt.plot(x_range,Gk,color='green')
x_range_list = []
x_range_list.append(x_range)
x_range_list.append(x_range)
GkList = []
GkList.append(Gk-sk)
GkList.append(Gk+sk)
plt.plot(x_range_list, GkList, color='orange')
plt.ylabel('G(k)+/- s_k')
plt.xlabel('number of clusters - k')
plt.show()

Gk = map(operator.sub, Elog_Wu.T, np.log(np.abs(W)))

# Implementation of the rule for estimating K*, see ESL (14.39), p. 519

K_opt = np.where(np.array(Gk[:-1]) >= np.array(map(operator.sub, Gk[1:], sk[1:])))[0]

if not K_opt.size:
    K_opt = k
    print("Gap-statistic, optimal K = %d" % K_opt)
else:
    print("Gap-statistic, optimal K = %d" % K_opt[0])

# Uncomment to see which cluster x_i belongs to

if K_opt[0] > 6:
    k = 6
else:
    k = K_opt[0]

kMeansOpt = KMeans(n_clusters=k, random_state=0).fit(X)
C = kMeansOpt.cluster_centers_ # the cluster means
resp = kMeansOpt.labels_ # a vector of the cluster number for each observation


fig = plt.figure()
for i in range(min(6,len(C))):
    c=C[i].reshape(16,16)
    c=np.rot90(c,1)
    plt.pcolor(c)
    plt.title(['Cluster %d' %i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

for i in range(1,401):
    #plt.subplot(4,4,16)
    c=np.array(X[i]).reshape(16,16)
    c=np.rot90(c,1)
    plt.pcolor(c)
    chat = resp[i]
    plt.title('Observation %d, cluster %d' %(i,chat))
    plt.gca().set_aspect('equal', adjustable='box')
    raw_input("Press Enter to continue...")
    plt.show()
