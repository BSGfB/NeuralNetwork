#include <stdio.h>

#include "MultiBackPropErrorLayer.hpp"

namespace NeuralNetwork {
    namespace Layer {   

        MultiBackPropErrorLayer::MultiBackPropErrorLayer() {
        }

        MultiBackPropErrorLayer::MultiBackPropErrorLayer(const MultiBackPropErrorLayer& orig) {
        }

        MultiBackPropErrorLayer::~MultiBackPropErrorLayer() {
        }
        
        ActivationFunction::AbstractActivationFunction* MultiBackPropErrorLayer::getActivationFunction() {
            return activationFunction;
        }

        float MultiBackPropErrorLayer::getSpeedLearning() {
            return speedLearning;
        }

        vector<float> MultiBackPropErrorLayer::getThresholdVector() {
            return thresholdVector;
        }

        vector<vector<float> > MultiBackPropErrorLayer::getWeightMatrix() {
            return weightMatrix;
        }

        void MultiBackPropErrorLayer::setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) {
            this->activationFunction = activationFunction;
        }

        void MultiBackPropErrorLayer::setSpeedLearning(float speedLearning) {
            this->speedLearning = speedLearning;
        }

        void MultiBackPropErrorLayer::setWeightMatrix(int inputSize, int outputSize) {
            this->weightMatrix.resize(outputSize, vector<float>(inputSize, 0.0001f));
            this->thresholdVector.resize(outputSize, 0.f);

            this->input.resize(inputSize, 0.f);
            this->output.resize(outputSize, 0.f);
            this->errorVector.resize(outputSize, 0.f);
        }
        
        
        
        void MultiBackPropErrorLayer::adjust() {
            for(int i = 0; i < weightMatrix.size(); i++) {
                float temp = speedLearning * activationFunction->getDerivativeValue(output[i]) * errorVector[i];
                thresholdVector[i] += temp; 
                
                for(int j = 0; j < weightMatrix[i].size(); j++) {
                    weightMatrix[i][j] -= temp * input[j];
                }
            }
        }

        std::vector<float> MultiBackPropErrorLayer::computeBackwardError(std::vector<float> error) {
            this->errorVector = error;
            vector<float> nextErrorVector(weightMatrix[0].size(), 0.0f);
            
            for(int i = 0; i < weightMatrix.size(); i++) {
                float temp = error[i] * activationFunction->getDerivativeValue(output[i]);
                
                for(int j = 0; j < weightMatrix[i].size(); j++) {
                    nextErrorVector[j] += temp * weightMatrix[i][j];
                }
            }
            return nextErrorVector;
        }

        std::vector<float> MultiBackPropErrorLayer::computeOutput(std::vector<float> input) {
            this->input = input;
            vector<float> activatedVector(weightMatrix.size());
                        
            for(int i = 0; i < weightMatrix.size(); i++) {
                this->output[i] = 0.0f;
                
                for(int j = 0; j < weightMatrix[i].size(); j++) {
                    this->output[i] += weightMatrix[i][j] * input[j];
                }
                
                this->output[i] -= thresholdVector[i];
                activatedVector[i] = this->activationFunction->getValue(this->output[i]);
            }
            
            return activatedVector;
        }

        unsigned int MultiBackPropErrorLayer::getInputSize() {
            return weightMatrix.size() != 0 ? weightMatrix[0].size() : 0;  
        }

        unsigned int MultiBackPropErrorLayer::getOutputSize() {
             return weightMatrix.size();
        }
        
        void MultiBackPropErrorLayer::randomize(float min, float max) {
            srand(time(0));
            
            for(unsigned int i = 0; i < weightMatrix.size(); i++) {
                for(unsigned int j = 0; j < weightMatrix[i].size(); j++) {
                    weightMatrix[i][j] = min + (max - min) * ((float)rand() / (float)RAND_MAX);
                }
            }
        }
    }
}