#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
__global__ void cuda_gaussian1d(T *data, T *g, T *f, long src_size,
                                long src_col_size, long data_col_size,
                                long g_size) {
  long j = blockDim.x * blockIdx.x + threadIdx.x;
  long i = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < src_size) {
    if (j < src_col_size) {
      f[src_col_size * i + j] = 0;
      for (long k = 0; k < g_size; k++) {
        f[src_col_size * i + j] += data[data_col_size * i + j + k] * g[k];
      }
    }
  }
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

void set_block_thread(dim3 *grid, dim3 *block, long x_size, long y_size) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int threads_per_block = deviceProp.maxThreadsPerBlock;
  block->y = 2;
  block->x = threads_per_block / block->y;
  grid->y = ceil(y_size / (float)block->y);
  grid->x = ceil(x_size / (float)block->x);
}

template <typename T>
std::vector<std::vector<T>> gaussian1d_multi(std::vector<std::vector<T>> src,
                                             T truncate, T sd) {
  long r = (long)(truncate * sd + 0.5);

  long row_size = src.size();
  long column_size = src[0].size();
  long data_column_size = column_size + 2 * r;

  std::vector<std::vector<T>> data(row_size, std::vector<T>(data_column_size));

  for (long n = 0; n < row_size; n++)
    data[n] = complement_data(src[n], r);

  // Gaussian distribution
  std::vector<T> gauss = gaussian_kernel(r, sd);

  // Filtered data
  std::vector<std::vector<T>> f(row_size, std::vector<T>(column_size));
  T *d_data, *d_gauss, *d_f;
  cudaMalloc((void **)&d_data, sizeof(T) * row_size * data_column_size);
  cudaMalloc((void **)&d_gauss, sizeof(T) * gauss.size());
  cudaMalloc((void **)&d_f, sizeof(T) * f.size() * column_size);
  cudaMemcpy(d_gauss, gauss.data(), sizeof(T) * gauss.size(),
             cudaMemcpyHostToDevice);
  for (int i = 0; i < row_size; i++)
    cudaMemcpy(d_data + data_column_size * i, data[i].data(),
               sizeof(T) * data_column_size, cudaMemcpyHostToDevice);

  dim3 grid, block;
  set_block_thread(&grid, &block, column_size, row_size);

  cuda_gaussian1d<<<grid, block>>>(d_data, d_gauss, d_f, row_size, column_size,
                                   data_column_size, gauss.size());
  cudaDeviceSynchronize();

  for (int i = 0; i < row_size; i++)
    cudaMemcpy(f[i].data(), d_f + column_size * i, sizeof(T) * column_size,
               cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_gauss);
  cudaFree(d_f);
  cudaDeviceReset();

  return f;
}

template <typename T>
std::vector<T> gaussian1d(std::vector<T> src,
                                             T truncate, T sd) {
  std::vector<std::vector<T>> v(1, std::vector<T>(src.size()));
  v[0] = src;
  return gaussian1d_multi(v, truncate, sd).at(0);
}

template std::vector<std::vector<float>>
gaussian1d_multi(std::vector<std::vector<float>> src, float truncate, float sd);
template std::vector<std::vector<double>>
gaussian1d_multi(std::vector<std::vector<double>> src, double truncate,
                 double sd);
template std::vector<float>
gaussian1d(std::vector<float> src, float truncate,
                                       float sd);
template std::vector<double>
gaussian1d(std::vector<double> src, double truncate,
                                       double sd);