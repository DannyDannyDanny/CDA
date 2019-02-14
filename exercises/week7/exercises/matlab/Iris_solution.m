% Gaussian Mixture Modelling
clear all;
close all

T = readtable('../Data/FisherIris.csv');
X = T{:,1:4};

figure(1)
plotmatrix(X)

for k=1:10
    disp(k)
    obj = gmdistribution.fit(X,k,'SharedCov',false,'Regularize',0.01,'Options',statset('MaxIter',1000)); 
    BIC(k)= obj.BIC;
end

figure(2)
clf
plot(BIC)
ylabel('BIC')
xlabel('# clusters')
 