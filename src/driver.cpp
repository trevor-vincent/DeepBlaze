#include <iostream>
#include "./blaze/math/dense/DynamicVector.h"
#include "./blaze/math/dense/DynamicMatrix.h"
#include "./blaze/math/TransposeFlag.h"
#include "./blaze/math/StorageOrder.h"
#include "./Optimizers/Optimizer.hpp"
#include "./Optimizers/StochGradDescent.hpp"

using Vector = blaze::DynamicVector<double,blaze::columnVector>;

int main(int argc, char *argv[])
{
  DeepBlaze::StochGradDescent sgd;
  Vector v1(25,1.0);
  Vector v2(25,2.0);
  std::cout << v2 << std::endl;
  sgd.update(v1,v2);
  std::cout << v2 << std::endl;
  return 0;
}

