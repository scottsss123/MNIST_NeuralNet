// --------------- includes ---------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

// --------------- constants ---------------------------------------------
const std::string training_images_filepath = "./train-images.idx3-ubyte";
const std::string training_labels_filepath = "./train-labels.idx1-ubyte";

const int n_pixels = 28 * 28;
const double learning_rate = 1;

// --------------- global variables ---------------------------------------------
std::vector<std::vector<unsigned char>> images;
std::vector<std::vector<unsigned char>> labels;

// random double gen, e.g. rand_real(e2);
std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> rand_real(-1, 1);
std::uniform_real_distribution<> small_rand_real(-0.1, 0.1);

// activation arrays 
std::vector<double> a0(n_pixels);
std::vector<double> a1(16);
std::vector<double> a2(16);
std::vector<double> a3(10);

// weight arrays
std::vector<double> w1(a0.size() * a1.size());
std::vector<double> w2(a1.size() * a2.size());
std::vector<double> w3(a2.size() * a3.size());

// bias arrays
std::vector<double> b1(16);
std::vector<double> b2(16);
std::vector<double> b3(10);

// tweak arrays    tweaks to weights/biases in layer 0/1/2/3
std::vector<double> tw1(a0.size() * a1.size());
std::vector<double> tw2(a1.size() * a2.size());
std::vector<double> tw3(a2.size() * a3.size());
std::vector<double> tb1(a1.size());
std::vector<double> tb2(a2.size());
std::vector<double> tb3(a3.size());

// activation 'tweaks'
std::vector<double> at2(a2.size());
std::vector<double> at1(a1.size());

int current_img_index = 1;

// --------------- functions ---------------------------------------------
std::vector<std::vector<unsigned char>> readImages(const std::string& filePath) {
	std::ifstream file(filePath, std::ios::binary);

	std::cout << "loading " << filePath << "\n";

	char magic_number_bytes[4];
	char n_images_bytes[4];
	char n_rows_bytes[4];
	char n_cols_bytes[4];

	file.read(magic_number_bytes, 4);
	file.read(n_images_bytes, 4);
	file.read(n_rows_bytes, 4);
	file.read(n_cols_bytes, 4);

	int n_images = (static_cast<unsigned char>(n_images_bytes[0]) << 24) | (static_cast<unsigned char>(n_images_bytes[1]) << 16) | (static_cast<unsigned char>(n_images_bytes[2]) << 8) | static_cast<unsigned char>(n_images_bytes[3]);
	int n_rows = (static_cast<unsigned char>(n_rows_bytes[0]) << 24) | (static_cast<unsigned char>(n_rows_bytes[1]) << 16) | (static_cast<unsigned char>(n_rows_bytes[2]) << 8) | static_cast<unsigned char>(n_rows_bytes[3]);
	int n_cols = (static_cast<unsigned char>(n_cols_bytes[0]) << 24) | (static_cast<unsigned char>(n_cols_bytes[1]) << 16) | (static_cast<unsigned char>(n_cols_bytes[2]) << 8) | static_cast<unsigned char>(n_cols_bytes[3]);

	std::cout << "n_images: " << n_images << "\nn_rows: " << n_rows << "\nn_cols: " << n_cols << "\n";

	std::vector<std::vector<unsigned char>> images;

	for (int i = 0; i < n_images; i++) {
		std::vector<unsigned char> image(n_rows * n_cols);

		file.read((char*)(image.data()), n_rows * n_cols);
		
		images.push_back(image);
	}

	file.close();
	return images;
}

std::vector<std::vector<unsigned char>> readLabels(const std::string& filePath) {
	std::ifstream file(filePath, std::ios::binary);

	std::cout << "loading " << filePath << "\n";

	char magic_number_bytes[4];
	char n_labels_bytes[4];

	file.read(magic_number_bytes, 4);
	file.read(n_labels_bytes, 4);

	int n_labels = (static_cast<unsigned char>(n_labels_bytes[0]) << 24) | (static_cast<unsigned char>(n_labels_bytes[1]) << 16) | (static_cast<unsigned char>(n_labels_bytes[2]) << 8) | static_cast<unsigned char>(n_labels_bytes[3]);

	std::cout << "n_labels: " << n_labels << "\n";

	std::vector<std::vector<unsigned char>> labels;

	for (int i = 0; i < n_labels; i++) {
		std::vector<unsigned char> label(1);

		file.read((char*)(label.data()), 1);

		labels.push_back(label);
	}

	file.close();
	return labels;
}

void initialiseWeights() {
	for (int i = 0; i < w1.size(); i++) {
		w1[i] = rand_real(e2);
	}
	for (int i = 0; i < w2.size(); i++) {
		w2[i] = rand_real(e2);
	}
	for (int i = 0; i < w3.size(); i++) {
		w3[i] = rand_real(e2);
	}
}

void initialiseBiases() {
	for (int i = 0; i < b1.size(); i++) {
		b1[i] = small_rand_real(e2);
	}
	for (int i = 0; i < b2.size(); i++) {
		b2[i] = small_rand_real(e2);
	}
	for (int i = 0; i < b3.size(); i++) {
		b3[i] = small_rand_real(e2);
	}
}

void populateInputs(int index) {
	for (int i = 0; i < n_pixels; i++) {
		double v = (double)images[index][i];
		a0[i] = v / 255; // normalises greyscale between 0 and 1
	}
}

void resetTweakArrs() {
	for (int i = 0; i < tw1.size(); i++) 
		tw1[i] = 0;
	for (int i = 0; i < tw2.size(); i++) 
		tw2[i] = 0;
	for (int i = 0; i < tw3.size(); i++)
		tw3[i] = 0;

	for (int i = 0; i < tb1.size(); i++)
		tb1[i] = 0;
	for (int i = 0; i < tb2.size(); i++)
		tb2[i] = 0;
	for (int i = 0; i < tb3.size(); i++)
		tb3[i] = 0;

	//for (int i = 0; i < at0.size(); i++)
	//	at0[i] = 0;
	for (int i = 0; i < at1.size(); i++)
		at1[i] = 0;
	for (int i = 0; i < at2.size(); i++)
		at2[i] = 0;
	//for (int i = 0; i < at3.size(); i++)
	//	at3[i] = 0;
}

int getLabel(int index) {
	return (int)labels[index][0];
}

double sigmoid(double x) {
	return 1 / (1 + std::exp(-x));
}

double inverse_sigmoid(double x) {
	return std::log(x / (1-x));
}

void outputVec(std::vector<double>& v) {
	for (int i = 0; i < v.size(); i++) {
		std::cout << v[i] << ", ";
	}
	std::cout << "\n";
}

void forwardProp() {
	// pixels -> layer 1
	for (int i = 0; i < a1.size(); i++) {
		double sum = 0;
		for (int j = 0; j < a0.size(); j++) {
			sum += a0[j] * w1[i * a0.size() + j];
		}
		sum += b1[i];
		a1[i] = sigmoid(sum);
	}
	// layer 1 -> layer 2
	for (int i = 0; i < a2.size(); i++) {
		double sum = 0;
		for (int j = 0; j < a1.size(); j++) {
			sum += a1[j] * w2[i * a1.size() + j];
		}
		sum += b2[i];
		a2[i] = sigmoid(sum);
	}
	// layer 2 -> layer 3
	for (int i = 0; i < a3.size(); i++) {
		double sum = 0;
		for (int j = 0; j < a2.size(); j++) {
			sum += a2[j] * w3[i * a2.size() + j];
		}
		sum += b3[i];
		a3[i] = sigmoid(sum);
	}
}

void updateWeightsAndBiases() {
	for(int i = 0; i < tw3.size(); i++) {
		//std::cout << "weight before: " << w3[i] << "\n";
		//std::cout << "tw3[i]: " << tw3[i] << "\n";
		w3[i] -= learning_rate * tw3[i]; // subtracting subtracting delCdelW => most quick decrease of cost
		//std::cout << "weight after: " << w3[i] << "\n";
	}
	for(int i = 0; i < tb3.size(); i++) {
		b3[i] -= learning_rate * tb3[i];
	}
	for(int i = 0; i < tw2.size(); i++) {
		w2[i] -= learning_rate * tw2[i];
	}
	for(int i = 0; i < tb2.size(); i++) {
		b2[i] -= learning_rate * tb2[i];
	}
}

void backProp() { // adds to tweak arrays -delC / del x : x = weights & biases for currently loaded image
	std::vector<double> y(10);
	y[getLabel(current_img_index)] = 1;

	// adds to tweak arrays how the cost most quickly decreases for current image
	// output layer
	for (int j = 0; j < a3.size(); j++) {
		//std::cout << "backprop last layer, node " << j << ":\n";
		for (int k = 0; k < a2.size(); k++) {
			//std::cout << "backprop penultimate layer, node " << k << ": ";
			// weights
			double delZdelW; // d(weighted sum) / d(weight)
			double delAdelZ; // d(activation of jth node in last layer) / d(weighted sum)
			double delCdelA; // d(cost) / d(activation of jth node in last layer)

			delCdelA = 2 * (a3[j] - y[j]);
			//delAdelZ = sigmoid(inverse_sigmoid(a3[k])) * (1 - sigmoid(inverse_sigmoid(a3[k])));
			delAdelZ = a3[k] * (1 - a3[k]);  
			delZdelW = a2[k];

			double delCdelW = delZdelW * delAdelZ * delCdelA;

			std::cout << "delCdelW: " << delCdelW << "\n"; // e-319 !?!?!?!?
			tw3[j * a2.size() + k] += delCdelW;

			// biases
			// delCdelB = delZdelB * delAdelZ * delCdelA
			//          = 1 * delAdelZ * delCdelA
			double delCdelB = 1 * delAdelZ * delCdelA;
			tb3[j] += delCdelB;

			// layer 2's optimal activaiton changes
			// delCdela = delZdela * delAdelZ * delCdelA
			double delZdela = w3[j * a2.size() + k];
			double delCdela = delZdela * delAdelZ * delCdelA;
			at2[k] += delCdela;
		}
	}
	for (int j = 0; j < a2.size(); j++) {
		for (int k = 0; k < a1.size(); k++) {
			double delZdelW = a1[k];
			double delAdelZ = a2[k] * (1 - a2[k]);
			double delCdelA = at2[j];

			double delCdelW = delZdelW * delAdelZ * delCdelA;
			tw2[j * a2.size() + k] += delCdelW;

			double delCdelB = 1 * delAdelZ * delCdelA;;
			tb2[j * a2.size() + k] += delCdelB;


		}
	}
	for (int i = 0; i < tw3.size(); i++) {
		tw3[i] *= (1 / a3.size());
	}
	for (int i = 0; i < tb3.size(); i++) {
		tb3[i] *= (1 / a3.size());
	}
	for (int i = 0; i < tw2.size(); i++) {
		tw2[i] *= (1 / a2.size());
	}	
	for (int i = 0; i < tb2.size(); i++) {
		tb2[i] *= (1 / a2.size());
	}
}

double cost(int target) {
	double sum = 0;
	std::vector<double> target_arr(a3.size());
	target_arr[target] = 1;

	// squared error
	for (int i = 0; i < a3.size(); i++) {
		sum += std::pow((target_arr[i] - a3[i]), 2);
	}

	return sum;
}



// --------------- main ---------------------------------------------
int main()
{
	images = readImages(training_images_filepath);
	labels = readLabels(training_labels_filepath);

	initialiseWeights();
	initialiseBiases();

	std::cout << "random weights:\n";
	populateInputs(current_img_index);
	forwardProp();
	std::cout << "output: ";
	outputVec(a3);
	std::cout << "cost: " << cost(getLabel(current_img_index)) << "\n";

	for (int j = 0; j < 1; j++) {
		std::cout << "training iteration " << j+1 << "\n";

		//std::cout << "w3 before: ";
		//outputVec(w3);

		backProp();
		updateWeightsAndBiases();

		//std::cout << "w3 after: ";
		//outputVec(w3);

		populateInputs(current_img_index);
		forwardProp();
		std::cout << "output: ";
		outputVec(a3);
		std::cout << "cost: " << cost(getLabel(current_img_index)) << "\n";
	}
}

