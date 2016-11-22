/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Layer.hpp
 * Author: sergey
 *
 * Created on November 18, 2016, 11:07 PM
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

namespace NeuralNetwork {
    namespace Layer {     
        class Layer {
        public:
            virtual unsigned int getInputSize() = 0;
            virtual unsigned int getOutputSize() = 0;
            virtual std::vector<float> computeOutput(std::vector<float>) = 0;
        };
    }
}

#endif /* LAYER_HPP */

