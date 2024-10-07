#include "includes.h"

class NeuralNet {
public:
	int n_inputs;
	int n_layers;
	std::vector<std::vector<std::vector<float>>> weights;
};


// WEIGHTS   MATRIX  (n(inputs) * n(nodes in layer 1))     ACTIVATION OF INPUTS VECTOR 
//
//  w0,0, w0,1, ...                                        A0
//  w1,0, w1,1, ...										                     A1
//  ...
//  ...
//  ...
//  ...

// MATRIX MULT ^
//    LAYER 1 WEIGHTS                     L2  L3
//  [ [[w0,0,  w0,1...],[w1,0, w1,1...]], [], [] ]
//  Size of weights = number of layers in network - 1
//  Size of weights[i] = size of layer i ( can be multiplied by size of layer i + 1 )