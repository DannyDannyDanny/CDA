clear, close all

class  = csvread('../Data/ziplabel.csv');
X      = csvread('../Data/zipdata.csv');

[N p] = size(X) ;
minX = min(X,[],1); % data range min
maxX = max(X,[],1); % data range max

count = 1; % count the loop number
Kvec = 1:20; % number of clusters - you may change this
for K = Kvec
    disp(K)
    % perform K-means
    [gr,C] = kmeans(X,K);
    % gr gives a vector of the cluster number ('gr') for each observation
    % C gives the cluster means
 
    % Compute within-class dissimilarity given X (the data), C (the cluster
    % centers) and gr (the predicted cluster numbers)
    W(count) = 0;
    for ik = 1:K     
        Ik = find(gr==ik);
        dk = sum((X(Ik,:) - ones(size(Ik))*C(ik,:)).^2,2);
        Dk = sum(dk);
        W(count) = W(count) + Dk;  
    end

    % Gap-statistic
    % 20 simulations of data uniformly distributed over [X]
    Nsim =20;
    for j=1:Nsim
        % simulate uniformly distributed data
        Xu = ones(N,1)*minX + rand(N,p).*(ones(N,1)*maxX-ones(N,1)*minX);

        % perform K-means
        [gru,Cu] = kmeans(Xu,K);
        % Compute within-class dissimilarity for the simualted data
        % given Xu (the simulated data), Cu (the cluster centers for the
        % simulated data), and gru (the predicted cluster numbers for teh
        % simulated data).
        Wu(count,j) = 0;
        for ik = 1:K     
            Ik = find(gru==ik);
            dk = sum((Xu(Ik,:) - ones(size(Ik))*Cu(ik,:)).^2,2);
            Dk = sum(dk);
            Wu(count,j) = Wu(count,j) + Dk;  
        end 
    end
    % clear variables for for-loop
    clear cl
    % count the loop number
    count = count+1;
end
% compute expectation of simualted wihtin-class dissimilarities, and the
% standard errors for the error bars
Elog_Wu = mean(log(Wu),2); % expected within cluster scatter
sk = std(log(Wu),[],2)*sqrt(1+1/Nsim); % standard error sk' in (14.39)

% Plot the log within class scatters
figure; title('Within-class dissimilarity'), hold on
plot(Kvec,log(W),'o-'), hold on
plot(Kvec, Elog_Wu,'s-r')
ylabel('log W_k')
xlabel('number of clusters - k')
legend('observed','expected for simulation')

% Plot the Gap curve
figure; title('Gap curve'), hold on
Gk = Elog_Wu'-log(W);
plot(Kvec,Gk,'.-')
ylabel('G(K) +/- s_k')
xlabel('number of clusters - k')
hold on
plot([Kvec;Kvec],[Gk-sk';Gk+sk'],'+-b')

% Implementation of the rule for estimating K*, see ESL (14.39), p. 519
K_opt = find(Gk(1:end-1)>=(Gk(2:end)-sk(2:end)'));

if isempty(K_opt), K_opt = K(end); end

fprintf('Gap-statistic, optimal K = %i\n',K_opt(1))

%%Uncomment to see which cluster x_i belong to
figure
clf
if K_opt(1)>15, K = 14; else K = K_opt(1); end

[resp, C] = kmeans(X, K);

for i=1:min(15,size(C,1))
    subplot(4,4,i)
    c=reshape((C(i,:)'),16,16);
    c=rot90(c,1);
    pcolor(c)
    title(['Cluster ',num2str(i)]);
    set(gca,'xtick',[],'ytick',[])
    axis square
end
for i=1:400,
    subplot(4,4,16)
    c=reshape((X(i,:)'),16,16);
    c=rot90(c,1);
    pcolor(c)
    chat = resp(i);
    title(['Observation ',num2str(i),', cluster ',num2str(chat)]);
    set(gca,'xtick',[],'ytick',[])
    axis square 
    pause
end




