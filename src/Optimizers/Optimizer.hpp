#ifndef OPTIMIZER_H
#define OPTIMIZER_H 

#include "../blaze/Math.h"

namespace DeepBlaze
{

///
/// \defgroup Optimizers Optimization Algorithms
///

///
/// \ingroup Optimizers
///
/// The interface of optimization algorithms
///
class Optimizer
{
    protected:
  using Vector = blaze::DynamicVector<double,blaze::columnVector>;

    public:
        virtual ~Optimizer() {}

        ///
        /// Reset the optimizer to clear all historical information
        ///
        virtual void reset() {};

        ///
        /// Update the parameter vector using its gradient
        ///
        /// It is assumed that the memory addresses of `dvec` and `vec` do not
        /// change during the training process. This is used to implement optimization
        /// algorithms that have "memories". See the AdaGrad algorithm for an example.
        ///
        /// \param dvec The gradient of the parameter. Read-only
        /// \param vec  On entering, the current parameter vector. On exit, the
        ///             updated parameters.
        ///
        virtual void update(Vector& dvec, Vector& vec) = 0;
};

} // namespace DeepBlaze



#endif
