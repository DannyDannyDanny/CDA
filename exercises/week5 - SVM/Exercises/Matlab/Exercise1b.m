clear; close all; clc;

%
% load some overlapping 2D data
%
T = csvread('../Data/Synthetic2DOverlap.csv');

X = T(:,1:2);
Y = T(:,3);

% Try different values for the kernel parameter

KernelScale = .05;   % <--- YOUR CHOICE <<<

%
% Estimate model
%
svm = fitcsvm(X,Y,'Standardize',true,'KernelFunction','rbf',...
              'KernelScale',KernelScale,'BoxConstraint',Inf);
%
% Draw observations and decision line
%
figure
plot(X(Y==1,1),X(Y==1,2),'b*')
hold on
plot(X(Y==0,1),X(Y==0,2),'r*')
h = 0.1; % Mesh grid step size
[X1,X2]   = meshgrid(-8:h:6,-5:h:5);
[~,score] = predict(svm,[X1(:),X2(:)]);
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2));
[~, h] = contour(X1,X2,scoreGrid,[0 0]);
set(h,'linecolor','k')
hold off


