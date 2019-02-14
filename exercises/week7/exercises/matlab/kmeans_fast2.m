function [asgn, C] = kmeans_fast2(X, K, par)

[N,d] = size(X);
try 
    asgn = par.asgn; 
catch
    asgn = 1 + floor( K * rand(1,N) ) ;
end
try draw = par.draw ; catch draw = 0 ; end

% we need to compute the distance between data and cluster centers
% we divide into three terms, d1 = "X^2", d2 = -2 * C * X, d3 = "C^2" 
d1 = sum( X.^2 , 2 ) ;
oK = ones(K,1) ;
oN = ones(N,1) ;

asgn_old = zeros(1,N) ;
c=1;

while sum(abs(asgn_old-asgn))
    
    asgn_old = asgn ;
   
    resp = sparse(asgn,1:N,1,K,N) ;
    
    invNK = diag(1 ./ max( 1 , sum( resp, 2 ) ) ) ;
    
    C = invNK * resp * X ;
    
    d3 = sum( C.^2 , 2 ) ;
    
    D = (d1(:,oK))' - 2 * C * X' + d3(:,oN) ;
    
    [val,asgn] = min(D,[],1) ;
    
    if draw
        clf
        colors = 'r.b.g.m.k.' ;
        colorsx = 'rxbxgxmxkx' ;
        colorso = 'robogomoko' ;
        for k=1:K
            indxk = find(asgn == k) ;
            plot(X(indxk,1),X(indxk,2),colors(1+2*(k-1):2*k)),
            hold on;
            plot(C(k,1),C(k,2),colorsx(1+2*(k-1):2*k))
            plot(C(k,1),C(k,2),colorso(1+2*(k-1):2*k))
        end
        hold off
        drawnow
        bigfig
        pause
    end
   % print('-djpeg',sprintf('kmeans_%d',c))
  c=c+1;
end
