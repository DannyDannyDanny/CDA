function beta = lda(X, y)
% BETA = LDA(X,Y)
% Perform linear discriminant analysis on the input data X with class
% labels y. y must contain exactly two distinct labels.
%
% Outputs the coefficients of the discriminating hyperplane; in 2D this is
% beta(1)*x + beta(2)*y + beta(3) = 0. Multiplying beta with an observation
% will give a number > 0 iff it is correctly classified and it has the
% larger label of the two classes.
%
% Author: kas and lhc, 2011

n = size(X,1);

labels = unique(y);

if length(labels) ~= 2
  error('Too many labels, two required.');
end

X1 = X(y==labels(2),:);
X2 = X(y==labels(1),:);

n1 = size(X1,1);
n2 = size(X2,1);

% priors
pi_1 = n1/n;
pi_2 = n2/n;

% means
mu_1 = mean(X1);
mu_2 = mean(X2);

% common covariance
sigma_1 = cov(X1);
sigma_2 = cov(X2);
sigma = (sigma_1*(n1-1) + sigma_2*(n2-1))/(n-2);

p = sigma\(mu_1 - mu_2)';
q = log(pi_1/pi_2) - 0.5*(mu_1 + mu_2)/sigma*(mu_1 - mu_2)';

beta = [p' q];