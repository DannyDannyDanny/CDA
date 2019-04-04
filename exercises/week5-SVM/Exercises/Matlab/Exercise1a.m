clear; close all; clc;
%
% Load data
%
T = csvread('../Data/Synthetic2DNoOverlapp.csv');

X = T(:,1:2);
Y = T(:,3);

%
% Parameters to the Support Vector Machine
%
KernelFunction  = 'linear';   % <--- YOUR CHOICE, 'rbf', 'linear', 'polynomial'
KernelScale     = .1;      % <--- YOUR CHOICE, used when 'rbf'
PolynomialOrder = 2;       % <--- YOUR CHOICE, used when 'polynomial'
%
% Estimate model
%
switch KernelFunction
    case 'rbf'
        svm = fitcsvm(X,Y,'Standardize',true,...
              'KernelFunction',KernelFunction,'KernelScale',KernelScale,...
              'BoxConstraint',Inf);
    case 'polynomial'
             svm = fitcsvm(X,Y,'Standardize',true,...
              'KernelFunction',KernelFunction,'PolynomialOrder',PolynomialOrder,...
              'BoxConstraint',Inf);   
    case 'linear'
             svm = fitcsvm(X,Y,'Standardize',true,...
              'KernelFunction',KernelFunction,...
              'BoxConstraint',Inf);   
    otherwise
        disp('Unsupported kernel');
end
%
% Draw observations and decision line
%
figure
plot(X(Y==1,1),X(Y==1,2),'b*')
hold on
plot(X(Y==0,1),X(Y==0,2),'r*')
h = 0.1; 
[X1,X2] = meshgrid(-8:h:6,-5:h:5);
[~,score] = predict(svm,[X1(:),X2(:)]);
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2));
[~, h] = contour(X1,X2,scoreGrid,[0 0]);
set(h,'linecolor','k')
hold off


