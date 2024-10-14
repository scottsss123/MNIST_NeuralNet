#include "includes.h"

std::uniform_real_distribution<double> weightDistr(-1.0, 1.0);
std::default_random_engine random_engine;

double Sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}
std::vector<double> OneHot(int index, int len) {
	std::vector<double> out(len);
	for (int i = 0; i < len; i++) {
		if (i == index) {
			out[i] = 1.0;
			continue;
		} 
		out[i] = 0.0;
	}
	return out;
}

class Layer {
public:
	int numNodesIn, numNodesOut;
	std::vector<double> curActivations;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;
	
public:
	Layer(int nNodesIn, int nNodesOut) : numNodesIn(nNodesIn), numNodesOut(nNodesOut), curActivations(numNodesOut)
	{
		weights.resize(numNodesIn, std::vector<double>(numNodesOut));
		biases.resize(numNodesOut);
		curActivations.resize(numNodesOut);
		InitRandomWeights();
	}

	void InitRandomWeights() {
		for (int j = 0; j < weights.size(); j++) {
			for (int k = 0; k < weights[j].size(); k++) {
				weights[j][k] = weightDistr(random_engine);
			}
		}
		for (int j = 0; j < biases.size(); j++) {
			biases[j] = weightDistr(random_engine);
		}
	}

	void OutputWeights() {
		for (int j = 0; j < weights.size() j++) {
			for (int k = 0; k < weights[j].size(); k++) {
				std::cout << weights[j][k] << ", ";
			}
		}
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
		for (int i = 0; i < layerSizes.size() - 1; i++) {
			layers.emplace_back(Layer(layerSizes[i], layerSizes[i + 1]));
		}
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

	std::vector<double> ForwardPropagate(std::vector<double> inputs) {
		for (Layer layer : layers) {
			inputs = layer.CalculateOutputs(inputs);
		}
		SoftMaxify(inputs);
		return inputs;
	}

	void UpdateWeights(std::vector<std::vector<std::vector<double>>> updates) {
		for (int i = 0; i < updates.size(); i++) {
			for (int j = 0; j < updates[i].size(); j++) {
				for (int k = 0; k < updates[i][j].size(); k++) {
					layers[i].weights[j][k] += updates[i][j][k];
				}
			}
		}
	}

	void Train(std::vector<std::vector<double>>& images, std::vector<double>& labels, int imagesPerEpoch, int epochs) {
		int epochCount = 0;
		int imageIndex = 0;
		while (imageIndex < images.size() && epochCount < epochs) {
			// initialise weightUpdates with 0s * biases
			std::vector<std::vector<std::vector<double>>> weightUpdates(layers.size());
			std::vector<std::vector<double>> biasUpdates(layers.size());
			for (int i = 0; i < layers.size(); i++) {
				weightUpdates[i].resize(layers[i].weights.size());
				biasUpdates.resize(layers[i].numNodesOut);
			}
			for (int i = 0; i < layers.size(); i++) {
				for (int j = 0; j < weightUpdates[i].size(); j++) {
					weightUpdates[i][j].resize(layers[i].weights[j].size());
					for (int k = 0; k < weightUpdates[i][j].size(); k++) {
						weightUpdates[i][j][k] = 0.0;
					}
				}
				for (int j = 0; j < biasUpdates[i].size(); j++) {
					biasUpdates[i][j] = 0.0;
				}
			}

			// loop through images in epoch & update weightChanges accordingly
			for (int epochIndex = imageIndex; epochIndex < imageIndex + imagesPerEpoch; epochIndex++) {
				// image and label for this exaple
				std::vector<double> image = images[epochIndex];
				int label = labels[epochIndex];
				// put this shit through the network , make sure to store activations
				// find prediction arr
				// use one hot function with label
				double predArr(10);

				// create and resize delArr
				std::vector<std::vector<double>> delArr(layers.size());
				for (int i = 0; i < delArr.size(); i++) {
					delArr[i].resize(layers[i].numNodesOut);
				}
				// backprop
				// last layer del // later move into rest of network with if
				const int lastLayerIndex = layers.size() - 1;
				for (int i = 0; i < layers[lastLayerIndex].numNodesOut; i++) {
					double del = 2.0 * ()
				}
				// rest of network del
				for (int i = lastLayerIndex; i > - 2; i--) {  // last layer to input layer
					for (int j = 0; j < layers[i].weights.size(); j++) {  // 0th node to last node
						for (int k = 0; k < layers[i].weights[j].size(); k++) {  // ( 0th node to 0th node ) to (last node to last node)
							// layers[i].weights[j][k] = weight between jth node in layer i and kth node in layer i+1
							// calculate del;
							double del = 0;
							
						}
					}
				}
			}

			epochCount++;
			imageIndex += imagesPerEpoch;
		}
	}

	void OuputClassification(std::vector<double> inputs) {
		std::vector<double> outputs = ForwardPropagate(inputs);

		for (int i = 0; i < outputs.size(); i++) {
			std::cout << "[ " << i << ",  " << outputs[i] << " ]\n";
		}
	}
};