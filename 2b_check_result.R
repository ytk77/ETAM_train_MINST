library(dplyr)
library(Rcpp)
library(doMC)
library(tictoc)
sourceCpp("fnc_ETAM_a_neuron.cpp")
source("load_MNIST.R")
source("fnc_Process.R")
registerDoMC(24)



X.train = load_image_file("data/train-images-idx3-ubyte") %>% as.matrix()
y1 = load_label_file("data/train-labels-idx1-ubyte")
Y.train = matrix(0, nrow(X.train), n_distinct(y1))
for (ii in 1:length(y1)) {
    Y.train[ii, y1[ii]+1] = 1
}
# # names(X.y) = paste0("y=", seq(0,9))
XY.train = cbind(X.train, Y.train) %>% fnc_hardlim()

## check training accuracy
fn <- "result/mnist_W_2019-02-17 22:59:47.rds"
W <- readRDS(fn)

# X1 = XY.train %>% fnc_hardlim()
# x1 = head(XY.train, 1000)
X1 = XY.train
X1b = cbind(X1, -1)

for (ii in 1:1) {
    tic()
    y0 = W %*% t(X1b)
    Y1 = fnc_hardlim( t(y0) )
    X1b = cbind(Y1, -1)    
    toc()
    print( sum( abs(X1 - Y1)) )
}


## test if recall from Yhat = 0
Y0 <- Y.train * 0
XY0 <- cbind(X.train %>% fnc_hardlim(), Y0)
X1b = cbind(XY0, -1)
for (ii in 1:1) {
    tic()
    y0 = W %*% t(X1b)
    Y1 = fnc_hardlim( t(y0) )
    X1b = cbind(Y1, -1)    
    toc()
    print( sum( abs(X1 - Y1)) )
}

Yhat = Y1[, (ncol(Y1)-9):ncol(Y1)]
diff = (Yhat != Y.train) * 1
row.diff = rowSums(diff)
table(row.diff)
