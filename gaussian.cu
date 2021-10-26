#include <iostream>
#include <vector>
#include <numeric>

template <typename T>
void gaussian(std::vector<T> src, T truncate, T sd){
    long r = (long)(truncate * sd + 0.5);
    std::vector<T> data(src.size() + 2 * r);
    
    // Gaussian distribution
    std::vector<T> gauss(2 * r + 1);
    for(long i = -r; i <= r; i++)
        gauss[i + r] = exp(-0.5 * i * i / (sd * sd));
    T gauss_sum = std::accumulate(gauss.begin(), gauss.end(), (T)0);
    for(long i = 0; i < gauss.size(); i++)
        gauss[i] /= gauss_sum; // Normalization


    for(T e: gauss){
        std::cout << e << std::endl;
    }
}

int main(){
    std::vector<float> x(200);
    float truncate = 4.0;
    float sd = 1;
    gaussian<float>(x, truncate, sd);

    return 0;
}