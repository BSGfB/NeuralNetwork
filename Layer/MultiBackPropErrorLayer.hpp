/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MultiBackPropErrorLayer.hpp
 * Author: sergey
 *
 * Created on November 19, 2016, 9:10 AM
 */

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
            ActivationFunction::AbstractActivationFunction* getActivationFunction() const {
                return activationFunction;
            }

            float getSpeedLearning() const {
                return speedLearning;
            }

            vector<float> getThresholdVector() const {
                return thresholdVector;
            }

            void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) {
                this->activationFunction = activationFunction;
            }

            void setSpeedLearning(float speedLearning) {
                this->speedLearning = speedLearning;
            }

            vector<vector<float> > getWeightMatrix() const {
                return weightMatrix;
            }

            void setWeightMatrix(int inputSize, int outputSize) {
                this->weightMatrix.resize(outputSize, vector<float>(inputSize, 0.0001f));
                this->thresholdVector.resize(outputSize, 0.f);
                
                this->input.resize(inputSize, 0.f);
                this->output.resize(outputSize, 0.f);
                this->errorVector.resize(outputSize, 0.f);
            }
        };
    }
}
#endif /* MULTIBACKPROPERRORLAYER_HPP */

