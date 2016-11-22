#include "ExecutLayer.h"

namespace NeuralNetwork {
    namespace Layer { 
		ExecutLayer::ExecutLayer() {
			
		}

		ExecutLayer::~ExecutLayer() { 
		}
		
		void ExecutLayer::setWeightMatrix(vector< vector<float> > weightMatrix) {
			this->weightMatrix = weightMatrix;
		}
		
		vector< vector<float> > ExecutLayer::getWeightMatrix() {
			return this->weightMatrix;
		}

		void ExecutLayer::setThresholdVector(vector<float> thresholdVector) {
			this->thresholdVector = thresholdVector;
		}

		vector<float> ExecutLayer::getThresholdVector() {
			return this->thresholdVector;
		}

		unsigned int ExecutLayer::getInputSize() {
			return (this->weightMatrix.size() == 0? 0 :this->weightMatrix[0].size());
		}
        unsigned int ExecutLayer::getOutputSize() {
			return this->weightMatrix.size();
		}
        std::vector<float> ExecutLayer::computeOutput(std::vector<float> inputVector) {
			vector<float> outputVector(this->weightMatrix.size());

			for(int i = 0; i < this->weightMatrix.size(); i++) {
				for(int j = 0; j < this->weightMatrix[i].size(); j++) {
					outputVector[i] += this->weightMatrix[i][j] * inputVector[j];
				}
				outputVector[i] -= this->thresholdVector[i];
			}

			return outputVector;
		}
	}
}
