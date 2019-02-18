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

## stats on recall errors
diff1 <- abs(X1 - Y1)
row.diff = rowSums(diff1)
table(row.diff)

## get error patterns
idx <- which(row.diff>0)
X1b = X1[idx, ]
X2 <- cbind(X1b, -1)
Y2 <- W %*% t(X2) %>% t() %>% fnc_hardlim()
## which bit went wrong?
diff1 <- abs(X1b - Y2)
which(diff1[1, ]!=0)

ii = 493
x = X1b[1, ]
v1 <- W %*% matrix(c(x, -1))
x[ii]
v1[ii]

sourceCpp("fnc_ETAM_a_neuron.cpp")
w_theta = W[ii, ]
w = head(w_theta, length(w_theta)-1)
theta = tail(w_theta, 1)
test_pat_to_hyperplan_distance(ii-1, XY.train, w, theta)
test_pat_to_hyperplan_distance(ii-1, matrix(x, nrow=1), w, theta)


w1 = cpp_ETAM_learn_a_neuron(XY.train, ii, 0.0001)
w_theta = w1
w = w_theta[1, 1:(length(w_theta)-1)]
theta = w_theta[1, length(w_theta)]
test_pat_to_hyperplan_distance(ii-1, XY.train, w, theta)

# show_digit( head(X1b[1,],784) )
# show_digit( head(y1[3,],784) )

