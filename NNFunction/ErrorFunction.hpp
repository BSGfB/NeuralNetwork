/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ErrorFunction.hpp
 * Author: sergey
 *
 * Created on November 19, 2016, 8:39 PM
 */

#ifndef ERRORFUNCTION_HPP
#define ERRORFUNCTION_HPP

#include <vector>
#include "math.h"

using std::vector;

namespace NeuralNetwork {
    namespace NNFunction {  

        class ErrorFunction {
        public:
            static vector<float> getError(const vector<float>&, const vector<float>&);
            static float getSquareError(const vector<float>&, const vector<float>&);
        private:
            ErrorFunction();
            ErrorFunction(const ErrorFunction& orig);
        };
    }
}

#endif /* ERRORFUNCTION_HPP */

