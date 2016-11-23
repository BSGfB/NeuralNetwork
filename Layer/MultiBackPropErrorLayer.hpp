#ifndef MULTIBACKPROPERRORLAYER_HPP
#define MULTIBACKPROPERRORLAYER_HPP

#include "AbstractBackPropagationErrorLayer.hpp"
#include "../ActivationFunction/AbstractActivationFunction.hpp"

#include <vector>
#include <time.h>
#include <cstdlib>
#include <limits.h>

using std::vector;

namespace NeuralNetwork {
    namespace Layer {   
        
        class MultiBackPropErrorLayer : public AbstractBackPropagationErrorLayer {
        public:
            MultiBackPropErrorLayer();
            MultiBackPropErrorLayer(const MultiBackPropErrorLayer& orig);
            virtual ~MultiBackPropErrorLayer();
            
            void adjust() override;
            std::vector<float> computeBackwardError(std::vector<float>) override;
            std::vector<float> computeOutput(std::vector<float>) override;
            unsigned int getInputSize() override;
            unsigned int getOutputSize() override;
            void randomize(float, float) override;
             
            
        private:
            vector<vector<float>> weightMatrix;
            vector<float> thresholdVector;            
            ActivationFunction::AbstractActivationFunction *activationFunction;
            float speedLearning;
            
            vector<float> input;
            vector<float> output;
            vector<float> errorVector;
            
        public:
            ActivationFunction::AbstractActivationFunction* getActivationFunction() const;

            float getSpeedLearning() const;

            vector<float> getThresholdVector() const;

            void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction);

            void setSpeedLearning(float speedLearning);

            vector<vector<float> > getWeightMatrix() const;

            void setWeightMatrix(int inputSize, int outputSize);
        };
    }
}
#endif /* MULTIBACKPROPERRORLAYER_HPP */

