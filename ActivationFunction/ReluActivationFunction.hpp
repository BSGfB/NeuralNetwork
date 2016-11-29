/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ReluActivationFunction.hpp
 * Author: sergey
 *
 * Created on November 29, 2016, 10:52 PM
 */

#ifndef RELUACTIVATIONFUNCTION_HPP
#define RELUACTIVATIONFUNCTION_HPP

#include "AbstractActivationFunction.hpp"

namespace NeuralNetwork {
    namespace ActivationFunction {   
        class ReluActivationFunction : public AbstractActivationFunction {           
        public:
            ReluActivationFunction() {
            }
            
            ReluActivationFunction(const ReluActivationFunction& other) {
            }
            
            float getValue(float x) {              
                return x > 0 ? x : 0.01f * x;
            }
            
            float getDerivativeValue(float x) {
                return x > 0 ? 1.0f : 0.01f;
            }
            
            std::string getActivationFunctionName() override {
                return "ReluActivationFunction";
            }
        };  
        
    }
}

#endif /* RELUACTIVATIONFUNCTION_HPP */

