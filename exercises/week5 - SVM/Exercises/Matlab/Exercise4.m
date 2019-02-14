clear; close all;

%
% Load data
%
T = readtable('../Data/ACS.csv');

Itrain = T.Train==1;
Y_train = T.Y(Itrain);
Y_test  = T.Y(~Itrain);
X_train = T{Itrain,1:end-2};
X_test  = T{~Itrain,1:end-2};
%
% Logistic regression
%
B = glmfit(X_train,Y_train,'binomial');

Y_hat_lr = glmval(B,X_test,'logit') > 0.5;

AccuracyLogReg = sum(Y_hat_lr == Y_test)/length(Y_test);

fprintf('Accuracy by logistic regression = %f\n',AccuracyLogReg)
%
% Build a svm model.
% Use the fitcsvm function
%


