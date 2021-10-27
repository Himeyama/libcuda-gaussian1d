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

  return 0;
}