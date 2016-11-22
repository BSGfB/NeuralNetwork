/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   AbstractNeuralNetworkTeacher.hpp
 * Author: sergey
 *
 * Created on November 19, 2016, 7:07 PM
 */

#ifndef ABSTRACTNEURALNETWORKTEACHER_HPP
#define ABSTRACTNEURALNETWORKTEACHER_HPP

namespace NeuralNetwork {
    namespace Teacher {  
        class AbstractNeuralNetworkTeacher {
        public:
            virtual void run() = 0;
        };
    }
}

#endif /* ABSTRACTNEURALNETWORKTEACHER_HPP */

