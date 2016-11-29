/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MultiLayerNeuralNetworkExecutor.cpp
 * Author: sergey
 * 
 * Created on November 23, 2016, 12:54 AM
 */

#include "MultiLayerNeuralNetworkExecutor.hpp"

namespace NeuralNetwork {
    namespace Executor {  

        MultiLayerNeuralNetworkExecutor::MultiLayerNeuralNetworkExecutor() {
        }

        MultiLayerNeuralNetworkExecutor::MultiLayerNeuralNetworkExecutor(const MultiLayerNeuralNetworkExecutor& orig) {
        }

        MultiLayerNeuralNetworkExecutor::~MultiLayerNeuralNetworkExecutor() {
        }

        std::vector<float> MultiLayerNeuralNetworkExecutor::computeOutput(std::vector<float> input) {            
            for(int i = 0; i < layers.size(); i++) {
                input = layers[i]->computeOutput(input);
            }
            return input;
        }

        vector<Layer::AbstractMLPExecutLayer*> MultiLayerNeuralNetworkExecutor::getLayers() const {
            return layers;
        }

        void MultiLayerNeuralNetworkExecutor::setLayers(vector<Layer::AbstractMLPExecutLayer*> layers) {
            this->layers = layers;
        }    
    }
}