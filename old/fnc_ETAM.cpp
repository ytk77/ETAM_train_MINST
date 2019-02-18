#include <Rcpp.h>
using namespace Rcpp;

#define BIG_NUM 99999999.9


NumericMatrix normalize_W(NumericMatrix W)
{
    int N = W.ncol();
    for (int ii=0; ii<N; ii++) {
        double sum = 0.0;
        for (int jj=0; jj<N; jj++) {
            sum += fabs( W(ii, jj) );
        }
        for (int jj=0; jj<N; jj++) {
            W(ii, jj) = W(ii, jj)/sum;
        }
    }
    
    return(W);
}

void fnc_pat_to_hyperplan_distance(int ii,
                                   NumericMatrix &X, 
                                   NumericVector &w, NumericVector &theta,
                                   double &d_pos, double &d_neg, 
                                   int &idx_pos, int &idx_neg)
{
    int N = X.ncol();   // number of Neurons, columns/dimension of patterns
    int P = X.nrow();   // number of patterns
    d_pos = BIG_NUM;
    d_neg = -1*BIG_NUM;   // reset minimal distances
    
    for (int kk=0; kk<P; kk++) {  // compute every pattern's distance to hyperplane by W[ii, ]
        double dist = 0.0;
        for (int jj=0; jj<N; jj++) {
            dist += w(jj) * X(kk, jj);   // Eq. 8
        }
        dist -= theta(0);
        
        if (X(kk,ii) == 1) {
            if (dist < d_pos) {
                idx_pos = kk;
                d_pos = dist;
            }
        }
        
        if (X(kk,ii) == -1) {
            if (dist > d_neg) {
                idx_neg = kk;
                d_neg = dist;
            }
        }
    }
}


// [[Rcpp::export]]
List cpp_ETAM_learn_a_neuron(NumericMatrix X, int neuron_idx, double alpha)
{
    int ii = neuron_idx - 1;  // ii is index of neuron starting from 0
    int N = X.ncol();   // number of Neurons, columns/dimension of patterns
    int P = X.nrow();   // number of patterns
    NumericMatrix W(1, N);
    NumericVector theta(1);
    
    // Step 1. initialize weight matrix
    for (int jj=0; jj<N; jj++) {
        for (int kk=0; kk<P; kk++) {
            W(0, jj) += X(kk, ii) * X(kk, jj);  // Eq. 7
        }
    }
    // normalize W
    W = normalize_W(W);
    
    // Steps 3 to 8.
    double d_pos = BIG_NUM;
    double d_neg = -1*BIG_NUM;
    int idx_pos, idx_neg;
    
    Rcout << "train neuron " << ii+1 << std::endl;
    // Rcout << idx_pos << " : min. distance=" << d_pos << std::endl;
    // Rcout << idx_neg << " : min. distance=" << d_neg << std::endl;
    
    // Step 4. handle if all -1 or 1 patterns in ii'th dimension
    NumericVector x_ii = X(_, ii);   // Rcpp sugar syntax
    if ( is_true(all(x_ii == 1)) ) {
        theta(0) = -sqrt(N) - 1;
    } else if ( is_true(all(x_ii == -1)) ) {
        theta(0) = sqrt(N) + 1;
    } else  {
        // Eq. 8 & 9
        fnc_pat_to_hyperplan_distance(ii, X, W, theta, d_pos, d_neg, idx_pos, idx_neg);
        // Eq. 10
        theta(0) = theta(0) + (d_pos+d_neg)/2.0;
        // re-compute minimal distance
        fnc_pat_to_hyperplan_distance(ii, X, W, theta, d_pos, d_neg, idx_pos, idx_neg);
        
        // Step 5.
        double score1;
        score1 = (d_pos - d_neg)/2;
        Rcout << "  score of dist=" << score1 << " with " << d_pos << ", " << d_neg << std::endl;
        
        double score2;
        while (1==1) {
            NumericVector theta2 = theta;
            NumericVector w2 = W(0, _);
            for (int jj=0; jj<N; jj++) {
                // Rcout << w2(jj) << "-->";
                w2(jj) += alpha * (X(idx_pos, ii) * X(idx_pos, jj)+ X(idx_neg, ii) * X(idx_neg, jj));
                // Rcout << w2(jj) << std::endl;
            }
            
            fnc_pat_to_hyperplan_distance(ii, X, w2, theta2, d_pos, d_neg, idx_pos, idx_neg);
            theta2(0) = theta2(0) + (d_pos+d_neg)/2.0;
            fnc_pat_to_hyperplan_distance(ii, X, w2, theta2, d_pos, d_neg, idx_pos, idx_neg);
            
            score2 = (d_pos - d_neg)/2;
            Rcout << "  score dist=" << score2 << std::endl;
            
            if (score2 > score1 || d_pos<0 || d_neg>0) {
                score1 = score2;
                // normalize w2
                double sum = 0.0;
                for (int jj=0; jj<N; jj++) {
                    sum += fabs( w2(jj) );
                }
                for (int jj=0; jj<N; jj++) {
                    w2(jj) = w2(jj)/sum;
                }
                
                W(0, _) = w2;
                theta = theta2;
            } else {
                break;
            }
        }
    }
    
    List R1;
    R1["W"] = W;
    R1["theta"] = theta;
    return(R1);
}    

// [[Rcpp::export]]
void test_pat_to_hyperplan_distance(int ii,
                                   NumericMatrix X, 
                                   NumericVector w, NumericVector theta)
{
    int N = X.ncol();   // number of Neurons, columns/dimension of patterns
    int P = X.nrow();   // number of patterns
    double d_pos = BIG_NUM;
    double d_neg = -1*BIG_NUM;   // reset minimal distances
    int idx_pos, idx_neg;
    
    for (int kk=0; kk<P; kk++) {  // compute every pattern's distance to hyperplane by W[ii, ]
        double dist = 0.0;
        for (int jj=0; jj<N; jj++) {
            dist += w(jj) * X(kk, jj);   // Eq. 8
        }
        dist -= theta(ii);
        
        if (X(kk,ii) == 1) {
            if (dist < d_pos) {
                idx_pos = kk;
                d_pos = dist;
            }
        }
        
        if (X(kk,ii) == -1) {
            if (dist > d_neg) {
                idx_neg = kk;
                d_neg = dist;
            }
        }
    }
}
