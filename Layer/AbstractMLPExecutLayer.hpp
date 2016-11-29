/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   AbstractMLPExecutLayer.hpp
 * Author: sergey
 *
 * Created on November 27, 2016, 12:55 AM
 */

#ifndef ABSTRACTMLPEXECUTLAYER_HPP
#define ABSTRACTMLPEXECUTLAYER_HPP

#include "AbstractLayer.hpp"
#include <vector>

using std::vector;

namespace NeuralNetwork {
    namespace Layer {  
        class AbstractMLPExecutLayer : public AbstractLayer {
        public:
            virtual void setWeightMatrix(vector< vector<float> >) = 0;
            virtual void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction) = 0;
            virtual void setThresholdVector(vector<float>) = 0;
            
            virtual vector<float> getThresholdVector() = 0;
            virtual vector< vector<float> > getWeightMatrix() = 0;
            virtual ActivationFunction::AbstractActivationFunction* getActivationFunction() = 0;            
        };
    }
}

#endif /* ABSTRACTMLPEXECUTLAYER_HPP */

