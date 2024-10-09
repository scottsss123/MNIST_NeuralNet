#include "includes.h"

std::random_device rd;
std::mt19937 eng(rd());
std::uniform_real_distribution<double> weightDistr(-1.0, 1.0);

class Layer {
private:
	int numNodesIn, numNodesOut;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;

public:
	Layer(int nNodesIn, int nNodesOut) {
		numNodesIn = nNodesIn;
		numNodesOut = nNodesOut;

		weights.reserve(numNodesIn);
		for (auto row : weights) {
			row.reserve(numNodesOut);
		}
		biases.reserve(numNodesOut);
	}

	void InitRandomWeights() {
		for (auto row : weights) {
			for (int i = 0; i < row.size(); i++) {
				//row[i] = weightDistr(eng);
				row[i] = 0;
			}
		}
	}

	void OutputWeights() {
		for (auto row : weights) {
			for (int i = 0; i < row.size(); i++) {
				std::cout << row[i] << " ";
			}
			std::cout << "\n";
		}
	}

	double Sigmoid(double x) {
		double numerator = 1;
		double denominator = 1 + std::exp(-x);
		return numerator / denominator;
	}

	std::vector<double> CalculateOutputs(std::vector<double> inputs) {
		std::vector<double> weightedInputs(numNodesOut);

		for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
			double weightedInput = biases[nodeOut];
			for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
				weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
			}
			weightedInputs[nodeOut] = Sigmoid(weightedInput);
		}

		return weightedInputs;
	}
};

class NeuralNet {
private:
	std::vector<Layer> layers;
public:
	NeuralNet(std::vector<int> layerSizes) {
		layers.reserve(layerSizes.size() - 1);
		for (int i = 0; i < layers.size(); i++) {
			layers[i] = Layer(layerSizes[i], layerSizes[i + 1]);
		}
	}

	void InitRandomLayerWeights() {
		for (auto layer : layers) {
			layer.InitRandomWeights();
		}
	}

	void OutputLayerWeights() {
		int i = 0;
		for (auto layer : layers) {
			std::cout << "layer " << i << "\n";
			layer.OutputWeights();
			std::cout << "\n";
		}
	}

	std::vector<double> CalculateOutputs(std::vector<double> inputs) {
		std::vector<double> nextInputs = inputs;
		for (auto layer : layers) {
			nextInputs = layer.CalculateOutputs(nextInputs);
		}
		return nextInputs;
	}

	std::vector<double> Classify(std::vector<double> inputs) {
		std::vector<double> outputs = CalculateOutputs(inputs);
		
		double maxFound = - 1;
		int index = 0;
		for (int i = 0; i < outputs.size(); i++) {
			if (outputs[i] > maxFound) {
				maxFound = outputs[i];
				index = i;
			}
		}

		std::vector<double> classification = { (double)index, maxFound };
	}
};