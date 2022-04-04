#include <iostream>
#include <random>
#include <immintrin.h>
#include <sys/time.h>
using namespace std;
const int maxN = 6000;
float Data[maxN][maxN];
struct timeval start,ending;
int n = 20;
void init(){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            Data[i][j]=float (i+1) + float (j+1)*1.3;
        }
    }
}
void GaussElimi(){
    __m256 vt,va,vaji,vaik,vajk,vx;
    for (int i = 0; i < n - 1; ++i) {
        float xx=1.0/Data[i][i];
        vt = _mm256_set1_ps(xx);
        for (int j = i+1; j+8 <= n-1; j+=8) {
            va = _mm256_load_ps(Data[i]+j);
            va = _mm256_mul_ps(va,vt);
            _mm256_store_ps(Data[i]+j,va);
        }
        for (int j = n-n%8; j < n; ++j) {
            Data[i][j]/=Data[i][i];

        }
        Data[i][i] = 1.0;
        for (int j = i+1; j <= n-1; ++j) {
            vaji = _mm256_load_ps(Data[j]+i);
            for (int k = i+1; k+8 <= n-1 ; k+=8) {
                vaik = _mm256_load_ps(Data[i]+k);
                vajk = _mm256_load_ps(Data[j]+k);
                vx = _mm256_mul_ps(vaik,vaji);
                vajk = _mm256_sub_ps(vajk,vx);
                _mm256_store_ps(Data[j]+k,vajk);
            }
            for (int k = n-n%8; k < n; ++k) {
                Data[j][k] = Data[j][k] - Data[i][k]*Data[j][i];
            }
            Data[j][i] = 0.0;
        }
    }
}//对齐版本采用_mm256_load_ps,不对齐版本采用_mm256_load_ps
void display(){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout<<Data[i][j]<<" ";
        }
        cout<<endl;
    }
}
int main(){
    init();
    int step=10;
    int count=0;
    for (int i = 10; i <= 1000; i+=step) {
        n=i;
        gettimeofday(&start,NULL);
        gettimeofday(&ending,NULL);
        count=0;
        while(float (ending.tv_sec-start.tv_sec)<5){
            GaussElimi();
            gettimeofday(&ending,NULL);
            count++;
        }
        float x=float(ending.tv_sec-start.tv_sec);
        cout<<i<<","<<count<<','<<float(ending.tv_sec-start.tv_sec)<<","<<x/float(count)<<endl;
        if (n==100)
            step=100;
    }
    return 0;

}


