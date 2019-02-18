#include <Rcpp.h>
using namespace Rcpp;

#define BIG_NUM 99999999.9
#define N_MORE_TRAIN_AFTER_ALL_FIT 25


NumericMatrix normalize_a_weight(NumericMatrix W)
{
    int N = W.ncol();
    int ii = 0;
    
    double sum = 0.0;
    for (int jj=0; jj<N; jj++) {
        sum += fabs( W(ii, jj) );
    }
    for (int jj=0; jj<N; jj++) {
        W(ii, jj) = W(ii, jj)/sum;
    }

    return(W);
}

void fnc_pat_to_hyperplan_distance(int ii,
                                   NumericMatrix &X, 
                                   NumericVector &w, double theta,
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
        dist -= theta;
        
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
NumericMatrix cpp_ETAM_learn_a_neuron(NumericMatrix X, int neuron_idx, double alpha)
{
    int ii = neuron_idx - 1;  // ii is index of neuron starting from 0
    int N = X.ncol();   // number of Neurons, columns/dimension of patterns
    int P = X.nrow();   // number of patterns
    NumericMatrix W(1, N);
    double theta;
    
    // Step 1. initialize weight matrix
    for (int jj=0; jj<N; jj++) {
        for (int kk=0; kk<P; kk++) {
            W(0, jj) += X(kk, ii) * X(kk, jj);  // Eq. 7
        }
    }
    // normalize W
    W = normalize_a_weight(W);
    
    // Steps 3 to 8.
    double d_pos = BIG_NUM;
    double d_neg = -1*BIG_NUM;
    int idx_pos, idx_neg;
    
    Rcout << "train neuron " << neuron_idx << std::endl;
    // Rcout << idx_pos << " : min. distance=" << d_pos << std::endl;
    // Rcout << idx_neg << " : min. distance=" << d_neg << std::endl;
    
    // Step 4. handle if all -1 or 1 patterns in ii'th dimension
    NumericVector x_ii = X(_, ii);   // Rcpp sugar syntax
    if ( is_true(all(x_ii == 1)) ) {
        theta = -sqrt(N) - 1;
    } else if ( is_true(all(x_ii == -1)) ) {
        theta = sqrt(N) + 1;
    } else  {
        // Eq. 8 & 9
        fnc_pat_to_hyperplan_distance(ii, X, W, theta, d_pos, d_neg, idx_pos, idx_neg);
        // Eq. 10
        theta = theta + (d_pos+d_neg)/2.0;
        // re-compute minimal distance
        fnc_pat_to_hyperplan_distance(ii, X, W, theta, d_pos, d_neg, idx_pos, idx_neg);
        
        // Step 5.
        double score1;
        score1 = (d_pos - d_neg)/2;
        Rcout << "  score of dist=" << score1 << " with " << d_pos << ", " << d_neg << std::endl;
        
        double score2;
        int more_train_count=0;
        while (1==1) {
            double theta2 = theta;
            NumericVector w2 = W(0, _);
            for (int jj=0; jj<N; jj++) {
                // Rcout << w2(jj) << "-->";
                w2(jj) += alpha * (X(idx_pos, ii) * X(idx_pos, jj)+ X(idx_neg, ii) * X(idx_neg, jj));
                // Rcout << w2(jj) << std::endl;
            }
            
            fnc_pat_to_hyperplan_distance(ii, X, w2, theta2, d_pos, d_neg, idx_pos, idx_neg);
            theta2 = theta2 + (d_pos+d_neg)/2.0;
            fnc_pat_to_hyperplan_distance(ii, X, w2, theta2, d_pos, d_neg, idx_pos, idx_neg);
            
            score2 = (d_pos - d_neg)/2;
            Rcout << "  neuron "<< neuron_idx << " : score dist=" << score2 << std::endl;
            
            if (score2 > score1) {  // learning rate annealing
                alpha = alpha * 0.99999;
            }

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
            
            if (score2 > score1 || d_pos<0 || d_neg>0) {
                score1 = score2;
            } else if (d_pos>0 && d_neg<0) {
                more_train_count++;
                if (more_train_count > N_MORE_TRAIN_AFTER_ALL_FIT) {
                    Rcout << "done training. " << " with " << d_pos << ", " << d_neg << std::endl;
                    break;
                } else {
                    Rcout << " more training " << more_train_count << " times " << std::endl;
                }
            }
        }
    }
    
    NumericMatrix W2(1, N+1);
    for (int ii=0; ii<N; ii++) {
        W2(0, ii) = W(0, ii);
    }
    W2(0, N) = theta;
        
    return(W2);
}    


// [[Rcpp::export]]
void test_pat_to_hyperplan_distance(int ii,
                                   NumericMatrix X, 
                                   NumericVector w, double theta)
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
        dist -= theta;
        
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
    
    Rcout << "idx_pos=" << idx_pos << std::endl;
    Rcout << "d_pos=" << d_pos << std::endl;
    Rcout << "idx_neg=" << idx_neg << std::endl;
    Rcout << "d_neg=" << d_neg << std::endl;
    
}

