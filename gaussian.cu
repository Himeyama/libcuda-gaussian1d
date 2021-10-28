#include <iostream>
#include <numeric>
#include <vector>

#define THREADS_PER_BLOCK 512

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

template <typename T>
__global__ void cuda_gaussian1d_multi(T a
  // T *data, T *g, T *f, long src_size,
                                      // long src_col_size, long data_col_size,
                                      // long g_size
                                      ) {
  long i = blockDim.x * blockIdx.x + threadIdx.x;
  long j = blockDim.y * blockIdx.y + threadIdx.y;

  printf("%ld, %ld\n", i, j);

  // if (i < src_size) {
  //   if (j < src_col_size) {
  //     f[i * src_col_size + j] = 0;
  //     for (long k = 0; k < g_size; k++)
  //       f[i * src_col_size + j] += data[i * data_col_size + j + k] * g[k];
  //   }
  // }
}

long reflect_idx(long size, long i) {
  long p;
  p = (i % (size * 2)) - size;
  if (p < 0)
    p = -(p + 1);
  return p;
}

template <typename T> std::vector<T> gaussian_kernel(long r, T sd) {
  std::vector<T> gauss(2 * r + 1);
  T gauss_sum = 0;
  for (long i = -r; i <= r; i++)
    gauss_sum += gauss[i + r] = exp(-0.5 * i * i / (sd * sd));
  for (long i = 0; i < gauss.size(); i++)
    gauss[i] /= gauss_sum; // Normalization
  return gauss;
}

template <typename T>
std::vector<T> complement_data(std::vector<T> src, long r) {
  std::vector<T> data(src.size() + 2 * r);
  for (long i = 0; i < src.size(); i++)
    data[r + i] = src[i];
  for (long i = 0; i < r; i++)
    data[r - i - 1] = src[reflect_idx(src.size(), i + src.size())];
  for (long i = 0; i < r; i++)
    data[src.size() + r + i] = src[reflect_idx(src.size(), i)];
  return data;
}

template <typename T>
std::vector<std::vector<T>> gaussian1d_multi(std::vector<std::vector<T>> src,
                                             T truncate, T sd) {
  long r = (long)(truncate * sd * 0.5);

  std::vector<std::vector<T>> data(src.size(),
                                   std::vector<T>(src[0].size() + 2 * r));
  for (long n = 0; n < src.size(); n++) {
    data[n] = complement_data(src[n], r);
  }

  // Gaussian distribution
  std::vector<T> gauss = gaussian_kernel(r, sd);

  // Filtered data
  std::vector<std::vector<T>> f(src.size(), std::vector<T>(src[0].size()));
  T *gdata, *ggauss, *gf;
  cudaMalloc((void **)&gdata, sizeof(T) * data.size() * src[0].size());
  cudaMalloc((void **)&ggauss, sizeof(T) * gauss.size());
  cudaMalloc((void **)&gf, sizeof(T) * f.size() * src[0].size());
  cudaMemcpy(gdata, data.data(), sizeof(T) * data.size() * src[0].size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ggauss, gauss.data(), sizeof(T) * gauss.size(),
             cudaMemcpyHostToDevice);

  // dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  // dim3 grid(ceil(src.size() / (float)THREADS_PER_BLOCK),
  //           ceil(src[0].size() / (float)THREADS_PER_BLOCK));
  cuda_gaussian1d_multi<<<dim3(2, 2), dim3(1, 32)>>>(0.1
    // gdata, ggauss, gf, src.size(),
    //                                      src[0].size(), data[0].size(),
    //                                      gauss.size()
                                         );
  cudaDeviceSynchronize();
  T *filtered = (T *)malloc(sizeof(T) * f.size() * f[0].size());
  cudaMemcpy(filtered, gf, sizeof(T) * f.size() * f[0].size(),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < f.size(); i++)
    memcpy(f[i].data(), filtered + f[0].size() * i, sizeof(T) * f[0].size());
  free(filtered);

  cudaFree(gdata);
  cudaFree(ggauss);
  cudaFree(gf);
  cudaDeviceReset();

  return f;
}

template <typename T>
std::vector<T> gaussian1d(std::vector<T> src, T truncate, T sd) {
  long r = (long)(truncate * sd + 0.5);
  std::vector<T> data = complement_data(src, r);

  // Gaussian distribution
  std::vector<T> gauss = gaussian_kernel(r, sd);

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

  cuda_gaussian1d<<<ceil(src.size() / (float)THREADS_PER_BLOCK),
                    THREADS_PER_BLOCK>>>(gdata, ggauss, gf, src.size(),
                                         gauss.size());
  cudaDeviceSynchronize();
  cudaMemcpy(f.data(), gf, sizeof(T) * f.size(), cudaMemcpyDeviceToHost);

  cudaFree(gdata);
  cudaFree(ggauss);
  cudaFree(gf);
  cudaDeviceReset();

  return f;
}

template std::vector<float> gaussian1d(std::vector<float> src, float truncate,
                                       float sd);
template std::vector<double> gaussian1d(std::vector<double> src,
                                        double truncate, double sd);

template std::vector<std::vector<float>>
gaussian1d_multi(std::vector<std::vector<float>> src, float truncate, float sd);
template std::vector<std::vector<double>>
gaussian1d_multi(std::vector<std::vector<double>> src, double truncate,
                 double sd);
