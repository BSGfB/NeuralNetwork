#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

namespace NeuralNetwork {
    namespace Layer {     
        class AbstractLayer {
        public:
            virtual unsigned int getInputSize() = 0;
            virtual unsigned int getOutputSize() = 0;
            virtual std::vector<float> computeOutput(std::vector<float>) = 0;
        };
    }
}

#endif /* LAYER_HPP */

