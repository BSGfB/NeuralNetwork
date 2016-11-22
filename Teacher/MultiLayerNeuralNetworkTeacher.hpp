/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MultiLayerNeuralNetworkTeacher.hpp
 * Author: sergey
 *
 * Created on November 19, 2016, 7:11 PM
 */

#ifndef MULTILAYERNEURALNETWORKTEACHER_HPP
#define MULTILAYERNEURALNETWORKTEACHER_HPP

#include "AbstractNeuralNetworkTeacher.hpp"
#include "../Layer/AbstractBackPropagationErrorLayer.hpp"
#include "../NNFunction/ErrorFunction.hpp"

#include "../Other/MyLog.h"

#include <vector>
#include <string>

using std::vector;

namespace NeuralNetwork {
    namespace Teacher {
        
        class MultiLayerNeuralNetworkTeacher : public AbstractNeuralNetworkTeacher {
        public:
            MultiLayerNeuralNetworkTeacher();
            MultiLayerNeuralNetworkTeacher(const MultiLayerNeuralNetworkTeacher& orig);
            virtual ~MultiLayerNeuralNetworkTeacher();
                      
            void run() override;
            
            /* GET */
            int getCheckStep() const;
            vector<vector<float> > getInputDataSet() const;
            vector<Layer::AbstractBackPropagationErrorLayer*> getLayers() const;
            int getNumberOfIterations() const;
            vector<vector<float> > getOutputDataSet() const;
            float getTargetError() const;
            
            /* SET */
            void setCheckStep(int checkStep);
            void setInputDataSet(vector<vector<float> > inputDataSet);            
            void setLayers(vector<Layer::AbstractBackPropagationErrorLayer*> layers);            
            void setNumberOfIterations(int numberOfIterations);            
            void setOutputDataSet(vector<vector<float> > outputDataSet);            
            void setTargetError(float targetError);

        private:
            vector<Layer::AbstractBackPropagationErrorLayer*> layers;
            
            vector<vector<float>> inputDataSet;
            vector<vector<float>> outputDataSet;
            
            int checkStep;
            int numberOfIterations;
            float targetError;
             
        };
    }
}

#endif /* MULTILAYERNEURALNETWORKTEACHER_HPP */

