% Hierarchical clustering
clear, close all

class  = csvread('../Data/ziplabel.csv');
X      = csvread('../Data/zipdata.csv');

for i=1:400
    labels{i} = num2str(class(i));
end

[N p] = size(X) ;

% Set up cluster algorithm - try atlernatives for the two distance measures
d_sample = 'euclidean' ;    % sample-sample distance, type "help pdist" for alternatives
d_group = 'ward' ;          % group-group distance, type "help linkage" for alternatives. 
N_leafs = 10 ; % N ;        % number leafs, default in matlab is 30, also try N

% Form sample-sample distance matrix
Y = pdist( X , d_sample ); 

% Create hierarchical cluster tree with linkage function
Z = linkage( Y , d_group );

% Plot hierarchical cluster tree with dendrogram function
%figure
%[H, T] = dendrogram( Z , N_leafs,'labels',  labels' ); % standard matlab function
figure
[Info] = GSDendrogram( Z , labels , [] , N_leafs ); % with labels 

i = 1; % you may have a look at Info to get more infotmation about the dendrogram
Info{i,:,:} % this gives on the second place the label of the node we are in,
            % and on the third place it gives the observation labels which
            % ended in the given node.