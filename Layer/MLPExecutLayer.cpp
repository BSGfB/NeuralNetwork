/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MLPExecutLayer.cpp
 * Author: sergey
 * 
 * Created on November 27, 2016, 12:56 AM
 */

#include "MLPExecutLayer.hpp"


namespace NeuralNetwork {
    namespace Layer { 
        MLPExecutLayer::MLPExecutLayer() {
        }

        MLPExecutLayer::MLPExecutLayer(const MLPExecutLayer& orig) {
        }

        MLPExecutLayer::~MLPExecutLayer() {
        }

        void MLPExecutLayer::setWeightMatrix(vector<vector<float>> weightMatrix) {
            this->weightMatrix = weightMatrix;
        }

        vector< vector<float> > MLPExecutLayer::getWeightMatrix() {
            return this->weightMatrix;
        }

        void MLPExecutLayer::setThresholdVector(vector<float> thresholdVector) {
            this->thresholdVector = thresholdVector;
        }

        vector<float> MLPExecutLayer::getThresholdVector() {
            return this->thresholdVector;
        }

        unsigned int MLPExecutLayer::getInputSize() {
            return (this->weightMatrix.size() == 0? 0 :this->weightMatrix[0].size());
        }

        unsigned int MLPExecutLayer::getOutputSize() {
            return this->weightMatrix.size();
        }

        ActivationFunction::AbstractActivationFunction* MLPExecutLayer::getActivationFunction() {
            return activationFunction;
        }

        void MLPExecutLayer::setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) {
            this->activationFunction = activationFunction;
        }

        std::vector<float> MLPExecutLayer::computeOutput(std::vector<float> inputVector) {
            vector<float> outputVector(this->weightMatrix.size());
            
            for(unsigned int i = 0; i < outputVector.size(); i++) {
                if(inputVector.size() != weightMatrix[i].size()) {
                    printf("Error size! %u, %u\n", inputVector.size(), weightMatrix[i].size());
                }
                
                for(int j = 0; j < inputVector.size() && j < weightMatrix[i].size(); j++) {
                    outputVector[i] += this->weightMatrix[i][j] * inputVector[j];
                }
                
                outputVector[i] -= this->thresholdVector[i];
                outputVector[i] = activationFunction->getValue(outputVector[i]);
            }
            
            return outputVector;
        }
    }
}
