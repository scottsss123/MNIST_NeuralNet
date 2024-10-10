#include "includes.h"



class Layer {
public:
	int numNodesIn, numNodesOut;
	std::vector<double> curActivations;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;
	std::mt19937 eng;
	std::uniform_real_distribution<double> weightDistr;

public:
	Layer(int nNodesIn, int nNodesOut) : numNodesIn(nNodesIn), numNodesOut(nNodesOut), eng(std::random_device{}()), weightDistr(-1.0, 1.0), curActivations(numNodesOut)
	{
		weights.resize(numNodesIn, std::vector<double>(numNodesOut));
		biases.resize(numNodesOut);
		curActivations.resize(numNodesOut);
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

		for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
			double weightedInput = biases[nodeOut];
			for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
				weightedInput += inputs[nodeIn] * weights[nodeIn][nodeOut];
			}
			curActivations[nodeOut] = Sigmoid(weightedInput);
		}

		return curActivations;
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
		SoftMaxify(inputs);
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

	// inverse of sigmoid , to atain un-rectified activation of node
	double Logit(double x) {
		return std::log(x / (1 - x));
	}

	std::vector<double> calculateSingleWeightChanges(std::vector<double> inputs) {

	}

	void Train(std::vector<std::vector<double>>& images, std::vector<int>& labels) {
		// check is same number of images and labels
		int n_images = images.size();
		int n_labels = labels.size();
		if (n_images != n_labels) {
			std::cout << "n_images != n_labels\n";
			return;
		}
		else if (n_images == 0) {
			std::cout << "n_images == 0\n";
			return;
		}
		// find size of input
		int input_size = images[0].size();
		// store weight changes
		int n_weights = 0;
		int n_biases = 0;
		for (auto layer : layers) {
			n_weights += layer.numNodesIn * layer.numNodesOut;
			n_biases += layer.numNodesOut;
		}
		std::vector<double> weightChanges(n_weights + n_biases);
		

		
	}
};