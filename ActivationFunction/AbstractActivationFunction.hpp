/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   AbstractActivationFunction.hpp
 * Author: sergey
 *
 * Created on November 18, 2016, 10:39 PM
 */

#ifndef ABSTRACTACTIVATIONFUNCTION_HPP
#define ABSTRACTACTIVATIONFUNCTION_HPP

#include <string>

namespace NeuralNetwork {
    namespace ActivationFunction {       
        class AbstractActivationFunction {
            public:
                virtual float getValue(float) = 0;
                virtual float getDerivativeValue(float) = 0;
                virtual std::string getActivationFunctionName() = 0;
            };
    }
}

#endif /* ABSTRACTACTIVATIONFUNCTION_HPP */

