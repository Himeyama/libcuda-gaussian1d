#include <iostream>
#include <vector>
#include <numeric>

template <typename T>
void gaussian(std::vector<T> src, T truncate, T sd){
    long r = (long)(truncate * sd + 0.5);
    std::vector<T> data(src.size() + 2 * r);
    std::vector<T> gauss(2 * r + 1);
    for(long i = -r; i <= r; i++)
        gauss[i + r] = exp(-0.5 * i * i / (sd * sd));

    

    // for(T e: gauss){
    //     std::cout << e << std::endl;
    // }
}

int main(){
    std::vector<float> x(200);
    float truncate = 4.0;
    float sd = 1;
    gaussian<float>(x, truncate, sd);

    return 0;
}