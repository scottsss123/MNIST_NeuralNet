#include <iostream>
#include <vector>
#include <random>

// random float gen, dist(e2);
std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist(-1, 1); 

class Node {
public:
    float activation;
    float bias;
    std::vector<float> weights;


    Node(int n_inputs) {
        weights.resize(n_inputs);

        bias = dist(e2);
        for (int i = 0; i < weights.size(); i++) {
            weights[i] = dist(e2);
        }

        activation = 0;
    }
};

class Layer {
public:
    std::vector<Node> node_arr;


    Layer(int n_inputs, int layer_size) {
        for (int i = 0; i < layer_size; i++) {
            node_arr.push_back(Node(n_inputs));
        }
    }
};

class NeuralNet {
public:
    int n_inputs;
    std::vector<Layer> layer_arr;


    NeuralNet(int in_n_inputs, std::vector<int>& layer_sizes) {
        n_inputs = in_n_inputs;

        int layer_inputs = n_inputs;
        for (int i = 0; i < layer_sizes.size(); i++) {
            layer_arr.push_back(Layer(layer_inputs, layer_sizes[i]));
            layer_inputs = layer_sizes.at(i);
        }
    }
};

int main()
{
    std::vector<int> network_sizes = { 16,16,10 };
    NeuralNet network = NeuralNet(784, network_sizes);

    std::cout << network.layer_arr[0].node_arr[0].weights[0] << "\n";

    return 1;
}

