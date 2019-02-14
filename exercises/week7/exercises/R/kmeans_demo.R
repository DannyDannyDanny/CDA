###################################################
################K.MEANS_DEMO######################
##################################################
#SIMULATE DATA
#dimensions
p = 2 
#number of datapoints
N = 500  
#Number of mixture components
K = 3 
#Simulate data from a gaussian mixture distributions
components <- sample(1:K,prob=c(1/3,1/3,1/3),size=N,replace=TRUE)
muX1 <- 7*runif(K, 0, 1) #pick any random number, the higher it is the more
muX2<- 7*runif(K, 0, 1) #seperated the data will be. Here I tried 7
sds <- sqrt(c(1,1,1))
samplesX1 <-rnorm(n=N, mean = muX1[components], sd =sds[components])
samplesX2 <-rnorm(n=N, mean = muX2[components], sd =sds[components])

plot(samplesX1,samplesX2,col=c('red','#E69F00', '#56B4E9')[components])

##################################################################
#K.MEANS APPLIED ON THE SIMULATED DATA
X=cbind(samplesX1,samplesX2)
fit <- kmeans(X, 3, nstart = 20)
# get clusters means 
meanC=aggregate(X,by=list(fit$cluster),FUN=mean)
table(fit$cluster, components)
#plot the cluster means onto the photo
points(meanC$samplesX1,meanC$samplesX2)