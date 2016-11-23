#ifndef BACKPROPAGATIONERRORLAYER_HPP
#define BACKPROPAGATIONERRORLAYER_HPP

#include "AbstractLayer.hpp"
#include <vector>

namespace NeuralNetwork {
    namespace Layer {  
        class BackPropagationErrorLayer : public AbstractLayer {
        public:
            /*
             * Method sets start values between min and max
             */
            virtual void randomize(float, float) = 0;

            /*
             * Calculate error array in backwards.
             */
            virtual std::vector<float> computeBackwardError(std::vector<float>) = 0;
            
            /**
             * set new values of neural network for reducing the mean square error.
             */
            virtual void adjust() = 0;
            
        };
    }
}

#endif /* BACKPROPAGATIONERRORLAYER_HPP */

