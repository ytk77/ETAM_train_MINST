library(dplyr)
library(Rcpp)
sourceCpp("fnc_ETAM.cpp")
source("load_MNIST.R")
source("fnc_Process.R")


X.train = load_image_file("data/train-images-idx3-ubyte") %>% as.matrix()
y1 = load_label_file("data/train-labels-idx1-ubyte")
Y.train = matrix(0, nrow(X.train), n_distinct(y1))
for (ii in 1:length(y1)) {
    Y.train[ii, y1[ii]+1] = 1
}
# names(X.y) = paste0("y=", seq(0,9))
XY.train = cbind(X.train, Y.train) %>% fnc_hardlim()

R1 = cpp_ETAM_learning(XY.train, 0.0001)
saveRDS(R1, "res_MNIST.rds")

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
