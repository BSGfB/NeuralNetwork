#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <vector>

namespace NeuralNetwork {
    namespace Executor {     
        class AbstractExecutor {
        public:
            virtual std::vector<float> computeOutput(std::vector<float>) = 0;            
        };
    }
}

#endif /* LAYER_HPP */