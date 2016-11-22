/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ErrorFunction.cpp
 * Author: sergey
 * 
 * Created on November 19, 2016, 8:39 PM
 */

#include "ErrorFunction.hpp"

namespace NeuralNetwork {
    namespace NNFunction {  

        ErrorFunction::ErrorFunction() {
        }

        ErrorFunction::ErrorFunction(const ErrorFunction& orig) {
        }

        vector<float> ErrorFunction::getError(const vector<float>& firstVector, const vector<float>& secondVector) {
            vector<float> errorVector(firstVector.size());
            for(unsigned int i = 0; i < errorVector.size(); i++) { errorVector[i] = firstVector[i] - secondVector[i]; }
            return errorVector;
        }

        float ErrorFunction::getSquareError(const vector<float>& firstVector, const vector<float>& secondVector) {
            float squareError = 0.0f;
            for(unsigned int i = 0; i < firstVector.size(); i++) { squareError += (float)pow(firstVector[i] - secondVector[i], 2); }
            return squareError;
        }

    }
}