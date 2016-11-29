/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MultiLayerNeuralNetworkExecutor.hpp
 * Author: sergey
 *
 * Created on November 23, 2016, 12:54 AM
 */

#ifndef MULTILAYERNEURALNETWORKEXECUTOR_HPP
#define MULTILAYERNEURALNETWORKEXECUTOR_HPP

#include "AbstractExecutor.h"
#include "../Layer/NeuralNetworkLayer.h"

namespace NeuralNetwork {
    namespace Executor {  
        class MultiLayerNeuralNetworkExecutor : public AbstractExecutor {
        public:
            MultiLayerNeuralNetworkExecutor();
            MultiLayerNeuralNetworkExecutor(const MultiLayerNeuralNetworkExecutor& orig);
            virtual ~MultiLayerNeuralNetworkExecutor();
            
            std::vector<float> computeOutput(std::vector<float>) override;

            vector<Layer::AbstractMLPExecutLayer*> getLayers() const;

            void setLayers(vector<Layer::AbstractMLPExecutLayer*> layers);

        private:
            vector<Layer::AbstractMLPExecutLayer*> layers;
        };
    }
}

#endif /* MULTILAYERNEURALNETWORKEXECUTOR_HPP */

