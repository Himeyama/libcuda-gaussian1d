#ifndef GAUSSIAN1D_HPP
#define GAUSSIAN1D_HPP

#include <vector>
template <typename T>
std::vector<T> gaussian1d(std::vector<T> src, T truncate, T sd);

template <typename T>
std::vector<std::vector<T>> gaussian1d_multi(std::vector<std::vector<T>> src, T truncate, T sd);

#endif
