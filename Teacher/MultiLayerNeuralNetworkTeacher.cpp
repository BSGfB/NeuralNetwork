#include "MultiLayerNeuralNetworkTeacher.hpp"

namespace NeuralNetwork {
    namespace Teacher {  
        
        MultiLayerNeuralNetworkTeacher::MultiLayerNeuralNetworkTeacher() {
        }

        MultiLayerNeuralNetworkTeacher::MultiLayerNeuralNetworkTeacher(const MultiLayerNeuralNetworkTeacher& orig) {
        }

        MultiLayerNeuralNetworkTeacher::~MultiLayerNeuralNetworkTeacher() {
        }
        
        void MultiLayerNeuralNetworkTeacher::run() {
            for(unsigned long iteration = 0; iteration < numberOfIterations; iteration++) {
                for(unsigned int t = 0; t < inputDataSet.size(); t++) {
                    vector<float> currentData = inputDataSet[t];
                    for(unsigned int i = 0; i < layers.size(); i++) 
                        currentData = layers[i]->computeOutput(currentData);

                    vector<float> currentErrors = NNFunction::ErrorFunction::getError(currentData, outputDataSet[t]);
                    for(int i = layers.size() - 1; i >= 0; i--) {
                        currentErrors = layers[i]->computeBackwardError(currentErrors);
                    }
                    
                    for (unsigned int i = 0; i < layers.size(); i++)
                        layers[i]->adjust();
                }
                                
                if(iteration % checkStep == 0) {
                    float totalError = 0.0f;                           
                    
                    std::vector<Layer::ExecutLayer> exeLayers(layers.size());
                    for(int i = 0; i < exeLayers.size(); i++) {
                        exeLayers[i].setActivationFunction(layers[i]->getActivationFunction());
                        exeLayers[i].setThresholdVector(layers[i]->getThresholdVector());
                        exeLayers[i].setWeightMatrix(layers[i]->getWeightMatrix());
                    }
                    
                    for(unsigned int t = 0; t < inputDataSet.size(); t++) {
                        vector<float> currentData = inputDataSet[t];
                        for(unsigned int i = 0; i < layers.size(); i++) 
                            currentData = layers[i]->computeOutput(currentData);
                        
                        totalError += NNFunction::ErrorFunction::getSquareError(currentData, outputDataSet[t]);
                    }
                    
                    totalError *= 0.5;
                    
                    printf("\nIt: %u\tError: %f\n", iteration, totalError);
                    MyLog::addLog(LOG_TYPE::INFO, "It: " + std::to_string(iteration) + "\tError: " + std::to_string(totalError));

                    if(totalError <= targetError) {
                        return;
                    }
                }
            }                   
        }
        
        int MultiLayerNeuralNetworkTeacher::getCheckStep() const {
            return checkStep;
        }

        vector<vector<float> > MultiLayerNeuralNetworkTeacher::getInputDataSet() const {
            return inputDataSet;
        }

        vector<Layer::AbstractBackPropagationErrorLayer*> MultiLayerNeuralNetworkTeacher::getLayers() const {
            return layers;
        }
               

        float MultiLayerNeuralNetworkTeacher::getTargetError() const {
            return targetError;
        }

        vector<vector<float> > MultiLayerNeuralNetworkTeacher::getOutputDataSet() const {
            return outputDataSet;
        }

        int MultiLayerNeuralNetworkTeacher::getNumberOfIterations() const {
            return numberOfIterations;
        }
        
        void MultiLayerNeuralNetworkTeacher::setTargetError(float targetError) {
            this->targetError = targetError;
        }

        void MultiLayerNeuralNetworkTeacher::setOutputDataSet(vector<vector<float> > outputDataSet) {
            this->outputDataSet = outputDataSet;
        }

        void MultiLayerNeuralNetworkTeacher::setNumberOfIterations(int numberOfIterations) {
            this->numberOfIterations = numberOfIterations;
        }
        
        void MultiLayerNeuralNetworkTeacher::setLayers(vector<Layer::AbstractBackPropagationErrorLayer*> layers) {
            this->layers = layers;
        }

        void MultiLayerNeuralNetworkTeacher::setInputDataSet(vector<vector<float> > inputDataSet) {
            this->inputDataSet = inputDataSet;
        }

        void MultiLayerNeuralNetworkTeacher::setCheckStep(int checkStep) {
            this->checkStep = checkStep;
        }       
    }
}
