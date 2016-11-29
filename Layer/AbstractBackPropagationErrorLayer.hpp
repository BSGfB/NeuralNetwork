#ifndef BACKPROPAGATIONERRORLAYER_HPP
#define BACKPROPAGATIONERRORLAYER_HPP

#include "AbstractLayer.hpp"
#include "../ActivationFunction/NeuralNetworkActivationFunction.h"
#include <vector>

namespace NeuralNetwork {
    namespace Layer {  
        class AbstractBackPropagationErrorLayer : public AbstractLayer {
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
            
            
            virtual float getSpeedLearning() = 0;            
            
            virtual std::vector<float> getThresholdVector() = 0;
            
            virtual std::vector<std::vector<float> > getWeightMatrix() = 0;
            
            virtual ActivationFunction::AbstractActivationFunction* getActivationFunction() = 0;
            
            virtual void setWeightMatrix(int inputSize, int outputSize) = 0;
            
            virtual void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) = 0;
            
            virtual void setSpeedLearning(float speedLearning) = 0;            
        };
    }
}

#endif /* BACKPROPAGATIONERRORLAYER_HPP */

