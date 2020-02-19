install.packages('loe','.',repos = "https://cloud.r-project.org")
.libPaths('.')  
library('loe')
library('MASS')
"make.distmat" <-
  function(X) {
    X <- t(X)
    return(sqrt(abs(sweep(sweep(-2*t(X)%*%X, 1, colSums(X*X), "+"), 2, colSums(X*X), "+"))))
  }

"Objt.SOE"<-
  function(vecx, cm,n,P,C=1){
    N <- n
    NC <- nrow(cm)
    X <- matrix(vecx,nrow=N,ncol=P)
    tmpD <- make.distmat(X)
    .C("SOEobjt",
       arg1 = as.double(tmpD),
       arg2 = as.integer(cm),
       arg3 = as.double( C ),
       arg4 = as.integer(N),
       arg5 = as.integer(NC),
       arg6 = as.double(0)
    )$arg6
  }

"Grad.SOEobjt"<-
  function(vecx, cm,n, P, C=1){
    N <- n
    NC <- nrow(cm)
    X <- matrix(vecx,nrow=N,ncol=P)
    D<- make.distmat(X)
    igrad <- rep(0,N*P)
    .C("SOEgrad", 
       arg1=as.double(igrad),
       arg2=as.double(X),
       arg3=as.double(D),
       arg4=as.integer(cm),
       arg5=as.double( C ),
       arg6=as.integer( N ),
       arg7=as.integer( P ),
       arg8=as.integer(NC)
    )$arg1
  }

"SOE" <-
  function(CM, N,p=2, c=0.1,maxit =1000,report=100, iniX = "rand",rnd=10000){
    if(iniX[1]=="rand"){
      iniX <- mvrnorm(N,rep(0,p),diag(p))
    }
    if(nrow(CM)>1000000){
      sid <- sample(1:nrow(CM))
      PCM <- CM[sid[1:rnd],]
      inivecx <- as.vector(iniX)
      result.opt <- optim(par=inivecx, fn = Objt.SOE, gr= Grad.SOEobjt,cm=PCM,n=N,P=p,C=c,method ="BFGS",control = list( trace = TRUE,REPORT=report,maxit=maxit))
      X <- matrix(result.opt$par,nrow=N)
      str = result.opt$value
      return(list(X=X,str=str))
    }else{
      inivecx <- as.vector(iniX)
      result.opt <- optim(par=inivecx, fn = Objt.SOE, gr= Grad.SOEobjt,cm=CM,n=N,P=p,C=c,method ="BFGS",control = list( trace = TRUE,REPORT=report,maxit=maxit))
      X <- matrix(result.opt$par,nrow=N)
      str = result.opt$value
      return(list(X=X,str=str))
    }
  }


quadruplets = read.csv('./dataForLOE.csv')
quadruplets = quadruplets +1
quadruplets = data.matrix(quadruplets)

result <- SOE(CM=quadruplets, N=max(quadruplets),p=2, c=1,maxit =5000,report=100,rnd = 500000)

write.csv(result$X,'./loeEmb.csv')

