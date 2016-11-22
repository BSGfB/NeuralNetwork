/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   BackPropagationErrorLayer.hpp
 * Author: sergey
 *
 * Created on November 18, 2016, 11:09 PM
 */

#ifndef BACKPROPAGATIONERRORLAYER_HPP
#define BACKPROPAGATIONERRORLAYER_HPP

#include "Layer.hpp"
#include <vector>

namespace NeuralNetwork {
    namespace Layer {  
        class BackPropagationErrorLayer : public Layer {
        public:
            /*
             * Method sets start values between min and max
             */
            virtual void randomize(float, float) = 0;

            /*
             * Calculate error array in backwards.
             */
            virtual std::vector<float> computeBackwardError(std::vector<float>) = 0;
            
            /**
             * set new values of neural network for reducing the mean square error.
             */
            virtual void adjust() = 0;
            
        };
    }
}

#endif /* BACKPROPAGATIONERRORLAYER_HPP */

