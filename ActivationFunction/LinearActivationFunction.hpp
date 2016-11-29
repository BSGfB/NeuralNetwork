/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LinearActivationFunction.hpp
 * Author: sergey
 *
 * Created on November 20, 2016, 6:05 PM
 */

#ifndef LINEARACTIVATIONFUNCTION_HPP
#define LINEARACTIVATIONFUNCTION_HPP

#include "AbstractActivationFunction.hpp"

namespace NeuralNetwork {
    namespace ActivationFunction {       
        class LinearActivationFunction : public AbstractActivationFunction {
            float getDerivativeValue(float s) override {
                return 1.0f;
            }
            
            float getValue(float s) override {
                return s;
            }
            
            std::string getActivationFunctionName() override {
                return "LinearActivationFunction";
            }
        };
        
    }
}

#endif /* LINEARACTIVATIONFUNCTION_HPP */

