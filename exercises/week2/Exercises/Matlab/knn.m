function error = knn(X_train,y_train,X_test,y_test,k)
% ERROR = KNN(X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,K)
% Get the error rate for the k-nearest neighbor classifier.
% X_train = training data set
% y_train = training labels
% X_test = test data set
% y_test = test labels
% k = number of classes
%
% Author: lhc, kas. 2011

n1 = size(X_train,1);
n2 = size(X_test,1);

errors = zeros(n2,length(k));
for i = 1:n2 % i runs over all data points in the test set
  d = sqrt(sum((X_train - ones(n1,1)*X_test(i,:)).^2,2)); % distance between test point i and training points  
  [dummy d_index] = sort(d); % sort the distances
  for kk = 1:length(k) % kk: running through elements of k (can run for multiple inputs)
    ind = d_index(1:k(kk)); % use the k nearest neighbors
    y_hat = round(mean(y_train(ind))); % estimate y_hat as the mean of y for knn
    errors(i,kk) = abs(y_test(i) - y_hat); % estimate error between true y and estimated y for test set
  end
end

error = sum(errors)/n2; % mean of the error made for each data point
