/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MLPExecutLayer.hpp
 * Author: sergey
 *
 * Created on November 27, 2016, 12:56 AM
 */

#ifndef MLPEXECUTLAYER_HPP
#define MLPEXECUTLAYER_HPP

#include "../NeuralNetwork.h"
#include <vector>

using std::vector;

namespace NeuralNetwork {
    namespace Layer {   
        
	class MLPExecutLayer : public AbstractMLPExecutLayer {
	private:
            vector<vector<float>> weightMatrix;
            vector<float> thresholdVector;
            ActivationFunction::AbstractActivationFunction *activationFunction;
            
        public:
            MLPExecutLayer();
            MLPExecutLayer(const MLPExecutLayer& orig);
            virtual ~MLPExecutLayer();
			
            unsigned int getInputSize() override;
            unsigned int getOutputSize() override;
            std::vector<float> computeOutput(std::vector<float>) override;

            void setWeightMatrix(vector< vector<float> >) override;
            void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) override;
            void setThresholdVector(vector<float>) override;
            
            vector<float> getThresholdVector() override;
            vector<vector<float>> getWeightMatrix() override;
            ActivationFunction::AbstractActivationFunction* getActivationFunction() override;
            
        };
    }
}

#endif /* MLPEXECUTLAYER_HPP */

