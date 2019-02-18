library(dplyr)
library(Rcpp)
library(doMC)
sourceCpp("fnc_ETAM_a_neuron.cpp")
source("load_MNIST.R")
source("fnc_Process.R")
registerDoMC(16)


X.train = load_image_file("data/train-images-idx3-ubyte") %>% as.matrix()
y1 = load_label_file("data/train-labels-idx1-ubyte")
Y.train = matrix(0, nrow(X.train), n_distinct(y1))
for (ii in 1:length(y1)) {
    Y.train[ii, y1[ii]+1] = 1
}
# names(X.y) = paste0("y=", seq(0,9))
XY.train = cbind(X.train, Y.train) %>% fnc_hardlim()


W = foreach(ii=1:ncol(XY.train), .combine=rbind) %dopar% {
    w1 = cpp_ETAM_learn_a_neuron(XY.train, ii, 0.00005)
}

fn = paste0("result/mnist_W_", 
            format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ".rds")
saveRDS(W, fn)

## check training accuracy
# R1 <- readRDS("res_MNIST.rds")
# x1 = XY.train
# 
# y0 = R1$W %*% t(x1) - R1$theta 
# y1 = fnc_hardlim( t(y0) )
# # show_digit( head(x1[1,],784) )
# show_digit( head(y1[1,],784) )
# 
# for (ii in 1:10) {
#     y0 = R1$W %*% t(y1) - R1$theta 
#     y1 = fnc_hardlim( t(y0) )
#     show_digit( head(y1[3,],784) )
#     Sys.sleep(0.5)    
# }
# 
# # sum(XY.train - y1)
# 
# c1 <- data.frame(A = XY.train[1,], B=y1[1,])
