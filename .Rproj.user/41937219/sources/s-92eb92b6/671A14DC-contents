#include <Rcpp.h>
using namespace Rcpp;


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define SETS 10
#define P 5	//The number of patterns.
#define N 10	//The number of neurons.
#define N1 11	//N+1
#define All1 1024	//2^N
#define NONE 9999


// [[Rcpp::export]]
NumericVector timesTwo(NumericVector x) {
  return x * 2;
}


float normalize(float w[N1])
{
    int i;
    float len;
    
    for (i=0, len=0; i<N; i++) len += w[i]*w[i];
    len = sqrt(len);
    for (i=0; i<N; i++) w[i] /= len;
    return len;
}




void main()
{
    
    //	int x[P][N1]={{1,1,1,-1,-1,-1,-1},{1,-1,1,1,-1,1,-1},
    //				{1,1,-1,1,-1,-1,-1}};
    int x[P][N1];
    //	int x[P][N1]={{-1,-1,-1,-1},{-1,-1,1,-1},{-1,1-1,-1},{-1,1,1,-1},
    //								{1,-1,-1,-1},{1,-1,1,-1},{1,1,-1,-1}};
    int recover[P];
    float w[N][N1], sum;
    float pmin, nmin, a=0.005;
    float premin, ppremin, prelen, pprelen, pret, ppret;
    int pid, nid, prepid, prenid, pprepid, pprenid;
    int result, bin[N1], pat;
    int i, j, k, l, n, temp, total=0, set;
    FILE *fp, *fp1, *fp2;
    int sp=0;
    int pattern[P];
    int DONE=0;
    
    fp = fopen("pp", "w");
    fp1 = fopen("data", "w");
    fp2 = fopen("INPUT.105", "r");
    
    
    
    for (set=0, total=0; set<SETS; set++) {
        for (k=0; k<P; k++) {
            fscanf(fp2, "%d", &temp);
            pattern[k]=temp;
            for (j=0; j<N; j++) x[k][j] = (temp>>j)%2 ? 1 : -1;
            x[k][N] = -1;
            recover[k] = NONE;
        }
        
        
        
        
        
        // new ///////////////////////////////////////////////////////////////////
        for (n=0; n<N; n++) {
            
            for (j=0; j<N; j++) {
                for (k=0, w[n][j]=0; k<P; k++) w[n][j] += x[k][n]*x[k][j];
            }
            w[n][N] = 0;
            normalize(w[n]);
            
            pmin = NONE;
            nmin = -NONE;
            for (k=0; k<P; k++) {
                for (j=0, sum=0; j<=N; j++) sum += w[n][j]*x[k][j];
                if (x[k][n]==1) {
                    if (sum<pmin) {
                        pmin = sum;		pid = k; } }
                else if (x[k][n]==-1) {
                    if (sum>nmin) {
                        nmin = sum;		nid = k; } }
            }
            
            
            
            // ///////////////////////////////////////////////
            premin = ppremin = -NONE;
            prepid = prenid = pprepid = pprenid = NONE;
            while (pmin!=NONE && nmin!=-NONE
                       //				&& ((pmin-nmin)/2>premin || premin>ppremin)) {
                       && ((pmin-nmin)/2>premin || !DONE)) {
                //				&& (pmin-nmin)/2>premin) {
                
                DONE = 1;
                pprepid = prepid;		pprenid = prenid;
                ppret = w[n][N];
                pprelen = prelen;
                ppremin = premin;
                
                prepid = pid;			prenid = nid;
                w[n][N] += (pmin+nmin)/2;
                for (j=0; j<N; j++)
                    w[n][j] += a*(x[pid][n]*x[pid][j]+x[nid][n]*x[nid][j]);
                prelen = normalize(w[n]);
                premin = (pmin-nmin)/2;
                
                
                pmin = NONE;
                nmin = -NONE;
                for (k=0; k<P; k++) {
                    for (j=0, sum=0; j<=N; j++) sum += w[n][j]*x[k][j];
                    if (x[k][n]==1) {
                        if (sum<0) DONE = 0;
                        if (sum<pmin) {
                            pmin = sum;		pid = k; } }
                    else if (x[k][n]==-1) {
                        if (sum>=0) DONE = 0;
                        if (sum>nmin) {
                            nmin = sum;		nid = k; } }
                }
            }
            
            // undo last change
            
            
            if (pmin!=NONE && nmin!=-NONE) {
                for (j=0; j<N; j++) {
                    w[n][j] *= prelen;
                    w[n][j] -= a*(x[prepid][n]*x[prepid][j]+x[prenid][n]*x[prenid][j]);
                    //w[n][j] *= pprelen;
                    //w[n][j] -= a*(x[pprepid][n]*x[pprepid][j]+x[pprenid][n]*x[pprenid][j]);
                }
                //w[n][N] = ppret;
            }
            else if (pmin==NONE) w[n][N] = 100;
            else if (nmin==-NONE) w[n][N] = -100;
        }
        fprintf(fp, "\n");
        
        
        
        
        // print weights //////////////////////////////////////////////////////////
        for (i=0; i<N; i++) {
            for (j=0; j<=N; j++) fprintf(fp, "%7.3f", w[i][j]);
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
        
        
        
        // write data /////////////////////////////////////////////////////////////
        for (k=0; k<All1; k++) {
            for (i=0; i<N; i++) bin[i] = (k>>i)%2 ? 1 : -1;
            bin[N] = -1;
            for (i=0, result=0; i<N; i++) {
                for (j=0, sum=0; j<=N; j++) sum += w[i][j]*bin[j];
                if (sum>0) result += pow(2, i);
                else if (sum==0 && bin[i]==1) result += pow(2, i);
            }
            fprintf(fp1, "%d\n", result);
        }
        
        
        
        // print result and recovery //////////////////////////////////////////////
        for (k=0, temp=0; k<P; k++) {
            
            for (i=0, pat=0; i<N; i++)
                if (x[k][i]==1) pat += pow(2, i);
                for (i=0, result=0; i<N; i++) {
                    for (j=0, sum=0; j<=N; j++) sum += w[i][j]*x[k][j];
                    if (sum>0) result += pow(2, i);
                    else if (sum==0 && x[k][i]==1) result += pow(2, i);
                    fprintf(fp, "%5.1f", sum);
                }
                fprintf(fp, "\n%d ---> %d\n", pat, result);
                if(pattern[k]==result){
                    sp++;
                }
                
                for (l=0; l<N; l++) {
                    x[k][l] = -x[k][l];
                    for (i=0, result=0; i<N; i++) {
                        for (j=0, sum=0; j<=N; j++) sum += w[i][j]*x[k][j];
                        if (sum>0) result += pow(2, i);
                        else if (sum==0 && x[k][i]==1) result += pow(2, i);
                    }
                    if (result==pat) temp++;
                    x[k][l] = -x[k][l];
                }
                
        }
        fprintf(fp, "\n\nrecover: %d/%d\n", temp, N*P);
        
        
        
        total += temp;
    }
    fprintf(fp, "\n\nrecover: %d/%d\n", total, N*P*10);
    
    printf("SP:%f\n",sp/10.0);
    printf("R: %f\n",total/10.0);
    
    
    fclose(fp1);
    fclose(fp);
    fclose(fp2);
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
timesTwo(42)
*/
