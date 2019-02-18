library(dplyr)
library(Rcpp)
sourceCpp("fnc_ETAM.cpp")

fnc_hardlim <- function(x)
{
    x = 1*(x>0)
    x[ x==0 ] = -1
    return(x)
}

N = 100
m1 = matrix(rnorm(N*800), ncol=N) 
pat = fnc_hardlim(m1)
pat = distinct( as.data.frame(pat) ) %>% as.matrix()
# print(pat)

R1 = cpp_ETAM_learning(pat, 0.001)

y0 = R1$W %*% t(pat) - R1$theta 
y1 = t(y0)
y1 = fnc_hardlim(y1)

sum(pat - y1)
