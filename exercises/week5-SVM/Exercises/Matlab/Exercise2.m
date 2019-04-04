clear; close all; clc;

% load funny data
load('manual_2D_ol_1.mat');

T = [X y];
csvwrite('Ex2Data.csv',T);

T = csvread('Ex2Data.csv');

X = T(:,1:2);
Y = T(:,3);
%
% Parameters to the Support Vector Machine
%
KernelFunction  = 'rbf';   % <--- YOUR CHOICE, 'rbf', 'linear', 'polynomial'
KernelScale     = .5;      % <--- YOUR CHOICE, used when 'rbf'
PolynomialOrder = 6;       % <--- YOUR CHOICE, used when 'polynomial'
BoxConstraint   = 20;      % <--- YOUR CHOICE, budget for margin
%
% Estimate model
%
switch KernelFunction
    case 'rbf'
        svm = fitcsvm(X,Y,'Standardize',true,...
              'KernelFunction',KernelFunction,'KernelScale',KernelScale,...
              'BoxConstraint',BoxConstraint);
    case 'polynomial'
             svm = fitcsvm(X,Y,'Standardize',true,...
              'KernelFunction',KernelFunction,'PolynomialOrder',PolynomialOrder,...
              'BoxConstraint',BoxConstraint);   
    case 'linear'
             svm = fitcsvm(X,Y,'Standardize',true,...
              'KernelFunction',KernelFunction,...
              'BoxConstraint',BoxConstraint);   
    otherwise
        disp(' Unsupported kernel');
end

%
% Draw observations and decision line
% Draw support vectors and support points
%
figure
plot(X(Y==1,1),X(Y==1,2),'b.')
hold on
plot(X(Y==-1,1),X(Y==-1,2),'r.')
h = 0.1; 
[X1,X2] = meshgrid(-10:h:10,-10:h:10);
[~,score] = predict(svm,[X1(:),X2(:)]);
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2));
[~, h] = contour(X1,X2,scoreGrid,[0 0]);
set(h,'linecolor','k','linewidth',2)
[~, h] = contour(X1,X2,scoreGrid,[1 1]);
set(h,'linecolor','k')
[~, h] = contour(X1,X2,scoreGrid,[-1 -1]);
set(h,'linecolor','k')
I = svm.IsSupportVector;
Xactive = X(I,:);
Isupportpoint = svm.Alpha>1e-6 & svm.Alpha<(BoxConstraint-1e-6);
plot(Xactive(Isupportpoint,1), Xactive(Isupportpoint,2), 'ks');
hold off


