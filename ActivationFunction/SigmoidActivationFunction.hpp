/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SigmoidActivationFunction.hpp
 * Author: sergey
 *
 * Created on November 18, 2016, 10:53 PM
 */

#ifndef SIGMOIDACTIVATIONFUNCTION_HPP
#define SIGMOIDACTIVATIONFUNCTION_HPP

#include "math.h"
#include "AbstractActivationFunction.hpp"

namespace NeuralNetwork {
    namespace ActivationFunction {   
        class SigmoidActivationFunction : public AbstractActivationFunction {           
        public:
            SigmoidActivationFunction() {
            }
            
            SigmoidActivationFunction(const SigmoidActivationFunction& other) {
            }
            
            float getValue(float x) {              
                return 1.0f / (1.0f + expf(-x));
            }
            
            float getDerivativeValue(float x) {
                float y = getValue(x);
                return y * (1.0f - y);
            }
            
            std::string getActivationFunctionName() override {
                return "SigmoidActivationFunction";
            }
        };  
        
    }
}

#endif /* SIGMOIDACTIVATIONFUNCTION_HPP */

