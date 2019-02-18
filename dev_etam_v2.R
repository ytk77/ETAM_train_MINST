library(dplyr)
library(Rcpp)
library(doMC)
sourceCpp("fnc_ETAM_a_neuron.cpp")
source("fnc_Process.R")
registerDoMC(16)


N = 10
m1 = matrix(rnorm(N*500), ncol=N) 
pat = fnc_hardlim(m1)
pat = distinct( as.data.frame(pat) ) %>% as.matrix()
# print(pat)

W = foreach(ii=1:N, .combine=rbind) %dopar% {
    w1 = cpp_ETAM_learn_a_neuron(pat, ii, 0.001)
}

p2 = cbind(pat, -1)  ## append a column for theta
y0 = W %*% t(p2) 
y1 = t(y0)
y1 = fnc_hardlim(y1)

sum(pat - y1)
