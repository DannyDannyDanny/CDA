function [resp, C] = kmeans_fast(X, K, par)

[N,d] = size(X);
meanX = mean(X);
try 
    par.covX; 
catch
    par.covX = (X-meanX(ones(N,1),:))'*(X-meanX(ones(N,1),:))/N;
end

% we need to compute the distance between data and cluster centers
% we divide into three terms, d1 = "X^2", d2 = -2 * C * X, d3 = "C^2" 
d1 = sum( X.^2 , 2 ) ;
oK = ones(K,1) ;
oN = ones(N,1) ;

C=randn(K,d)*chol(par.covX)+ones(K,1)*mean(X) ;

asgn_old = zeros(1,N) ;
asgn = ones(1,N) ;

while sum(abs(asgn_old-asgn))
    
    asgn_old = asgn ;
    
    d3 = sum( C.^2 , 2 ) ;
    
    D = (d1(:,oK))' - 2 * C * X' + d3(:,oN) ;
    
    [val,asgn] = min(D,[],1) ; 
    
    resp = sparse(asgn,1:N,1,K,N) ;
    
    invNK = diag(1 ./ max( 1 , sum( resp, 2 ) ) ) ;
    
    C = invNK * resp * X ;
  
end
