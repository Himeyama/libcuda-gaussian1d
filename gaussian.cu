#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
__global__ void cuda_gaussian1d(T *data, T *g, T *f, long src_size,
                                long g_size) {
  long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < src_size) {
    f[i] = 0;
    for (long j = 0; j < g_size; j++)
      f[i] += data[i + j] * g[j];
  }
}

long reflect_idx(long size, long i) {
  long p;
  p = (i % (size * 2)) - size;
  if (p < 0)
    p = -(p + 1);
  return p;
}

template <typename T>
std::vector<T> gaussian1d(std::vector<T> src, T truncate, T sd) {
  long r = (long)(truncate * sd + 0.5);

  std::vector<T> data(src.size() + 2 * r);
  for (long i = 0; i < src.size(); i++)
    data[r + i] = src[i];
  for (long i = 0; i < r; i++)
    data[r - i - 1] = src[reflect_idx(src.size(), i + src.size())];
  for (long i = 0; i < r; i++)
    data[src.size() + r + i] = src[reflect_idx(src.size(), i)];

  // Gaussian distribution
  std::vector<T> gauss(2 * r + 1);
  T gauss_sum = 0;
  for (long i = -r; i <= r; i++)
    gauss_sum += gauss[i + r] = exp(-0.5 * i * i / (sd * sd));
  for (long i = 0; i < gauss.size(); i++)
    gauss[i] /= gauss_sum; // Normalization

  // Filtered data
  std::vector<T> f(src.size());

  T *gdata, *ggauss, *gf;
  cudaMalloc((void **)&gdata, sizeof(T) * data.size());
  cudaMalloc((void **)&ggauss, sizeof(T) * gauss.size());
  cudaMalloc((void **)&gf, sizeof(T) * f.size());
  cudaMemcpy(gdata, data.data(), sizeof(T) * data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ggauss, gauss.data(), sizeof(T) * gauss.size(),
             cudaMemcpyHostToDevice);

  cuda_gaussian1d<<<(src.size() + 256) / 256, 256>>>(gdata, ggauss, gf,
                                                     src.size(), gauss.size());
  cudaDeviceSynchronize();
  cudaMemcpy(f.data(), gf, sizeof(T) * f.size(), cudaMemcpyDeviceToHost);

  return f;
}

template std::vector<float> gaussian1d(std::vector<float> src, float truncate, float sd);
template std::vector<double> gaussian1d(std::vector<double> src, double truncate, double sd);
