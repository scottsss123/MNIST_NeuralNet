#include "includes.h"



class Layer {
private:
	int numNodesIn, numNodesOut;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;
	std::mt19937 eng;
	std::uniform_real_distribution<double> weightDistr;

public:
	Layer(int nNodesIn, int nNodesOut) : numNodesIn(nNodesIn), numNodesOut(nNodesOut), eng(std::random_device{}()), weightDistr(-1.0, 1.0) 
	{
		weights.resize(numNodesIn, std::vector<double>(numNodesOut));
		biases.resize(numNodesOut);
		InitRandomWeights();
	}

	void InitRandomWeights() {
		for (auto& row : weights) {
			for (auto& weight : row) {
				weight = weightDistr(eng);
			}
		}
		for (auto& bias : biases) {
			bias = weightDistr(eng);
		}
	}

	void OutputWeights() {
		std::cout << "OUTPUT WEIGHTS\n";
		for (auto& row : weights) {
			for (auto& weight : row) {
				std::cout << weight << " ";
			}
			std::cout << "\n";
		}
	}

	double Sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	std::vector<double> CalculateOutputs(std::vector<double>& inputs) {
		std::vector<double> outputs(numNodesOut);

		for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
			double weightedInput = biases[nodeOut];
			for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
				weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
			}
			outputs[nodeOut] = Sigmoid(weightedInput);
		}

		return outputs;
	}
};

class NeuralNet {
private:
	std::vector<Layer> layers;
public:
	NeuralNet(std::vector<int>& layerSizes) {
		for (size_t i = 0; i < layerSizes.size() - 1; i++) {
			layers.emplace_back(layerSizes[i], layerSizes[i + 1]);
		}
	}

	void OutputLayerWeights() {
		for (size_t i = 0; i < layers.size(); i++) {
			std::cout << "Layer : " << i << "\n";
			layers[i].OutputWeights();
			std::cout << "\n";
		}
	}

	std::vector<double> CalculateOutputs(std::vector<double> inputs) {
		for (auto& layer : layers) {
			inputs = layer.CalculateOutputs(inputs);
		}
		return inputs;
	}

	void SoftMaxify(std::vector<double>& vec) {
		double sum = 0;
		for (int i = 0; i < vec.size(); i++) {
			sum += std::exp(vec[i]);
		}
		for (int i = 0; i < vec.size(); i++) {
			vec[i] = std::exp(vec[i]) / sum;
		}
	}

	std::pair<int, double> Classify(std::vector<double> inputs) {
		std::vector<double> outputs = CalculateOutputs(inputs);
		
		auto maxElement = std::max_element(outputs.begin(), outputs.end());
		int index = std::distance(outputs.begin(), maxElement);
		return { index, *maxElement };
	}

	void ClassifyAndOutput(std::vector<double>& inputs) {
		std::vector<double> outputs = CalculateOutputs(inputs);
		SoftMaxify(outputs);

		for (int i = 0; i < outputs.size(); i++) {
			std::cout << "[ " << i << ",  " << outputs[i] << " ]\n";
		}
	}

	double CalculateSingleCost(std::vector<double>& inputs, int label) {
		std::vector<double> predictions = CalculateOutputs(inputs);
		double sumError = 0;
		for (int i = 0; i < predictions.size(); i++) {
			double target = 0;
			if (i == label) {
				target = 1.0;
			}
			sumError += std::pow(predictions[i] - target, 2);
		}
		return sumError;
	}	
};