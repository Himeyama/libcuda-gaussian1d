#include <gaussian1d.hpp>
#include <iostream>
#include <vector>

int main() {
  std::vector<double> x{1, 2, 3, 4, 5};
  double truncate = 4.0;
  double sd = 4;
  std::vector<double> f = gaussian1d<double>(x, truncate, sd);
  for (double e : f) {
    std::cout << e << std::endl;
  }
  std::cout << std::endl;

  std::vector<std::vector<double>> xm = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  std::vector<std::vector<double>> fm =
      gaussian1d_multi<double>(xm, truncate, sd);
  for (auto e : fm) {
    for (auto f : e) {
      std::cout << f << ", ";
    }
    std::cout << std::endl;
  }

  return 0;
}