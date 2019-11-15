#ifndef STOCHGRADDESCENT_H
#define STOCHGRADDESCENT_H 

#include "../blaze/math/dense/DynamicVector.h"
#include "../blaze/math/dense/DynamicMatrix.h"
#include "../blaze/math/TransposeFlag.h"
#include "../blaze/math/StorageOrder.h"
#include "./Optimizer.hpp"


namespace DeepBlaze
{
  ///
  /// \ingroup Optimizers
  ///
  /// The Stochastic Gradient Descent algorithm
  ///
  class StochGradDescent: public Optimizer
  {
  private:
    using Matrix = blaze::DynamicMatrix<double,blaze::rowMajor>;
    using Vector = blaze::DynamicVector<double,blaze::columnVector>;

  public:
    double learning_rate_;
    double decay_;

    StochGradDescent() :
      learning_rate_(0.01), decay_(0)
    {}

    void update(Vector& dvec, Vector& vec)
    {
      vec -= learning_rate_ * (dvec + vec * decay_);
    }
  };


} // namespace DeepBlaze

#endif
