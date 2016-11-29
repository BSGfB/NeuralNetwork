#include <stdio.h>

#include "BackPropErrorLayer.hpp"

namespace NeuralNetwork {
    namespace Layer {   

        BackPropErrorLayer::BackPropErrorLayer() {
        }

        BackPropErrorLayer::BackPropErrorLayer(const BackPropErrorLayer& orig) {
        }

        BackPropErrorLayer::~BackPropErrorLayer() {
        }
        
        ActivationFunction::AbstractActivationFunction* BackPropErrorLayer::getActivationFunction() {
            return activationFunction;
        }

        float BackPropErrorLayer::getSpeedLearning() {
            return speedLearning;
        }

        vector<float> BackPropErrorLayer::getThresholdVector() {
            return thresholdVector;
        }

        vector<vector<float> > BackPropErrorLayer::getWeightMatrix() {
            return weightMatrix;
        }

        void BackPropErrorLayer::setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) {
            this->activationFunction = activationFunction;
        }

        void BackPropErrorLayer::setSpeedLearning(float speedLearning) {
            this->speedLearning = speedLearning;
        }

        void BackPropErrorLayer::setWeightMatrix(int inputSize, int outputSize) {
            this->weightMatrix.resize(outputSize, vector<float>(inputSize, 0.0001f));
            this->thresholdVector.resize(outputSize, 0.f);

            this->input.resize(inputSize, 0.f);
            this->output.resize(outputSize, 0.f);
            this->errorVector.resize(outputSize, 0.f);
        }
        
        void BackPropErrorLayer::adjust() {
            for(int i = 0; i < weightMatrix.size(); i++) {
                float temp = speedLearning * activationFunction->getDerivativeValue(output[i]) * errorVector[i];
                thresholdVector[i] += temp; 
                
                for(int j = 0; j < weightMatrix[i].size(); j++) {
                    weightMatrix[i][j] -= (temp * this->input[j]);
                }
            }
        }

        std::vector<float> BackPropErrorLayer::computeBackwardError(std::vector<float> error) {
            this->errorVector = error;
            vector<float> nextErrorVector(weightMatrix[0].size(), 0.0f);          
            
            for(int i = 0; i < weightMatrix.size(); i++) {
                float temp = error[i] * activationFunction->getDerivativeValue(output[i]);
                
                for(int j = 0; j < weightMatrix[i].size(); j++) {
                    nextErrorVector[j] += (temp * weightMatrix[i][j]);
                }
            }
            
            return nextErrorVector;
        }

        std::vector<float> BackPropErrorLayer::computeOutput(std::vector<float> input) {
            this->input = input;            
            std::vector<float> newOutput(this->output.size());                   
      
            for(int i = 0; i < weightMatrix.size(); i++) {
                this->output[i] = 0.0f;
                
                for(int j = 0; j < weightMatrix[i].size(); j++) 
                    this->output[i] += weightMatrix[i][j] * input[j];
                
                this->output[i] -= thresholdVector[i];
                newOutput[i] = this->activationFunction->getValue(this->output[i]);
            }
            
            return newOutput;
        }

        unsigned int BackPropErrorLayer::getInputSize() {
            return weightMatrix.size() != 0 ? weightMatrix[0].size() : 0;  
        }

        unsigned int BackPropErrorLayer::getOutputSize() {
             return weightMatrix.size();
        }
        
        void BackPropErrorLayer::randomize(float min, float max) {
            srand(time(0));
            
            for(unsigned int i = 0; i < weightMatrix.size(); i++) {
                for(unsigned int j = 0; j < weightMatrix[i].size(); j++) {
                    weightMatrix[i][j] = min + (max - min) * ((float)rand() / (float)RAND_MAX);
                }
            }
        }
    }
}