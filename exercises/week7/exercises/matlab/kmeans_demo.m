% k-means demo
close all

figure(1)
getdata; % simulate data set and plot in Figure 1

figure(2) % plot the demo in Figure 2
K = 2 ; % number of clusters - try different numbers
par.draw = 1 ;
[asgn,C] = kmeans_fast2(X,K,par) ; % run demo of k-means clustering - use "space" to proceed with next iteration

   