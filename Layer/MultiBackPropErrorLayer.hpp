#ifndef MULTIBACKPROPERRORLAYER_HPP
#define MULTIBACKPROPERRORLAYER_HPP

#include "../NeuralNetwork.h"

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
            float getSpeedLearning() override;
            vector<float> getThresholdVector() override;
            vector<vector<float> > getWeightMatrix() override;
            ActivationFunction::AbstractActivationFunction* getActivationFunction() override;

            void setWeightMatrix(int inputSize, int outputSize) override;
            void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) override;
            void setSpeedLearning(float speedLearning) override;
        };
    }
}
#endif /* MULTIBACKPROPERRORLAYER_HPP */

