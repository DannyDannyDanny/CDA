clear all
rand('seed',0); randn('seed',0);
p = 2 ;
N = 200 ;
oN = ones(N,1) ;

datatype = 'mixture' %normal'; %mixture'; %latent' ;
plottype = 'scatterlabel' %scatter'; % scatterlabel'; %histogram' ;

switch datatype
    case 'normal'
        spherical = 0 ; 
        mup = zeros(1,p)  ;
        if spherical 
            stdp = ones(1,p) ; 
            X = mup(oN,:) + stdp(oN,:) .* randn(N,p) ;
        else
            cov = [ 1 0.5 ; 0.5 1 ] ; % here it is assumed that p == 2
            covchol = chol(cov) ;
            X = mup(oN,:) + randn(N,p) * covchol ;
        end
    case 'mixture'
        K = 3 ;
        PK = ones(K,1) / K ;
        aPK = PK ; % accumulated distribution
        for i=2:K
            aPK(i) = aPK(i-1) + PK(i) ;
        end
        separation = 2.5 ; % 5 is well separateed and 2.5 not so well separated
        mup = separation * randn(K,p) ;  
        stdp = ones(K,p) ;
        X = zeros(N,p) ;
        for n = 1:N
            k = find( rand < aPK , 1 ) ; % select component
            label(n) = k ; 
            X(n,:) = mup(k,:) + stdp(k,:) .* randn(1,p) ;
        end
        prep = 0 ;
        if prep 
            meanX = mean(X) ;
            stdX = std(X) ;
            X = ( X - meanX(oN,:) ) ./ stdX(oN,:) ;
        end
    case 'latent'
        M = 3; % number of latent dimensions
        A = randn( M , p ) ;
        latenttype = 'pos'
        switch latenttype
            case 'normal'
                S = randn( N , M ) ;
            case 'poskur'
                S = randn( N , M ) ; S = S.^3 ;  %S.^4 .* sign( S ) ;       
            case 'negkur'
                S = rand( N , M ) ;
            case 'pos'
                A = abs( A ) ;
                S = randn( N , M ) ; S = S.^3 .* sign( S ) ; %S.^4 ;
        end
        X = S * A ; %+ randn( N , p ) ; % transposed notation compared to slides
end

switch plottype 
    case 'histogram'
        bins = 20 ;
        xmin = min( X ) ; xmax = max( X ) ;
        delta = ( xmax - xmin ) ;
        dbins = delta / ( bins - 3 ) ; 
        xval = xmin-dbins:dbins:xmax+dbins ;
        counts = histc(X,xval-0.5*dbins) ; % to center the counts
        bar(xval,counts/(N*dbins),1) ; hold on
        plot(X,zeros(N,1),'rx')
        plot(xval,normpdf(xval,mup,stdp))
        axis([ min(xval) max(xval) 0 1.1*max(counts/(N*dbins))]), hold off 
    case 'scatter'
        plot(X(:,1),X(:,2),'r.')
    case 'scatterlabel'
        colors = 'r.b.g.m.k.' ;
        for n=1:N
            plot(X(n,1),X(n,2),colors(1+2*(label(n)-1):2*label(n))), hold on;
        end
        hold off
end
            
bigfig
%print -depsc tmp.eps