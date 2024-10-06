#include "includes.h"

class NeuralNet {
public:

private:
	std::vector<std::vector<float>> values;
	std::vector<std::vector<float>> weights;
	std::vector < std::vector<float>> biases;

public:
	void init_weights();
private:
	float dotProduct(std::vector<float> A, std::vector<float> B);
	float reLU(float num);
	float sigmoid(float num);
	std::vector<float> softmax(std::vector<float> vec);
};