#ifndef EXECUTELAYER_H
#define EXECUTELAYER_H

#include <vector>
#include "AbstractLayer.hpp"
#include "../ActivationFunction/NeuralNetworkActivationFunction.h"

using std::vector;

namespace NeuralNetwork {
    namespace Layer {   
        
	class ExecutLayer : public AbstractLayer {
	private:
            vector< vector<float> > weightMatrix;
            vector<float> thresholdVector;
            ActivationFunction::AbstractActivationFunction *activationFunction;
            
        public:
            ExecutLayer();
            ~ExecutLayer();
			
            unsigned int getInputSize();
            unsigned int getOutputSize();
            std::vector<float> computeOutput(std::vector<float>);

            void setWeightMatrix(vector< vector<float> >);
            vector< vector<float> > getWeightMatrix();

            void setThresholdVector(vector<float>);
            vector<float> getThresholdVector();
            
            ActivationFunction::AbstractActivationFunction* getActivationFunction() const;

            void setActivationFunction(ActivationFunction::AbstractActivationFunction* activationFunction);

        };

    }
}

#endif