// --------------- includes ---------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

// --------------- constants ---------------------------------------------
const std::string training_images_filepath = "./train-images.idx3-ubyte";
const std::string training_labels_filepath = "./train-labels.idx1-ubyte";

const int n_pixels = 28 * 28;

// --------------- global variables ---------------------------------------------
std::vector<std::vector<unsigned char>> images;
std::vector<std::vector<unsigned char>> labels;

// random float gen, e.g. rand_real(e2);
std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> rand_real(-1, 1);
std::uniform_real_distribution<> small_rand_real(-0.1, 0.1);

// activation arrays 
std::vector<float> a0(n_pixels);
std::vector<float> a1(16);
std::vector<float> a2(16);
std::vector<float> a3(10);

// weight arrays
std::vector<float> w1(a0.size() * a1.size());
std::vector<float> w2(a1.size() * a2.size());
std::vector<float> w3(a2.size() * a3.size());

// bias arrays
std::vector<float> b1(16);
std::vector<float> b2(16);
std::vector<float> b3(10);

// tweak arrays    tweaks to weights/biases in layer 0/1/2/3
std::vector<float> tw1(a0.size() * a1.size());
std::vector<float> tw2(a1.size() * a2.size());
std::vector<float> tw3(a2.size() * a3.size());
std::vector<float> tb1(a1.size());
std::vector<float> tb2(a2.size());
std::vector<float> tb3(a3.size());

// activation 'tweaks'
std::vector<float> at3(a3.size());
std::vector<float> at2(a2.size());
std::vector<float> at1(a1.size());
std::vector<float> at0(a0.size());

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
		float v = (float)images[index][i];
		a0[i] = v / 255; // normalises greyscale between 0 and 1
	}
}

int getLabel(int index) {
	return (int)labels[index][0];
}

float sigmoid(float x) {
	return 1 / (1 + std::exp(-x));
}

float inverse_sigmoid(float x) {
	std::log(x / (1-x));
}

void outputVec(std::vector<float>& v) {
	for (int i = 0; i < v.size(); i++) {
		std::cout << v[i] << ", ";
	}
	std::cout << "\n";
}

void forwardProp() {
	// pixels -> layer 1
	for (int i = 0; i < a1.size(); i++) {
		float sum = 0;
		for (int j = 0; j < a0.size(); j++) {
			sum += a0[j] * w1[i * a0.size() + j];
		}
		sum += b1[i];
		a1[i] = sigmoid(sum);
	}
	// layer 1 -> layer 2
	for (int i = 0; i < a2.size(); i++) {
		float sum = 0;
		for (int j = 0; j < a1.size(); j++) {
			sum += a1[j] * w2[i * a1.size() + j];
		}
		sum += b2[i];
		a2[i] = sigmoid(sum);
	}
	// layer 2 -> layer 3
	for (int i = 0; i < a3.size(); i++) {
		float sum = 0;
		for (int j = 0; j < a2.size(); j++) {
			sum += a2[j] * w3[i * a2.size() + j];
		}
		sum += b3[i];
		a3[i] = sigmoid(sum);
	}
}

void backProp() {
	// adds to tweak arrays how the cost most quickly decreases for current image
	for (int j = 0; j < a3.size(); j++) {
		for (int k = 0; k < a2.size(); k++) {
			float wt = 0; // weight tweak

			float delZdelW; // d(weighted sum) / d(weight)
			float delAdelZ; // d(activation of jth node in last layer) / d(weighted sum)
			float delCdelA; // d(cost) / d(activation of jth node in last layer)

			delCdelA = 2 * (a3[j] - )

			tw3[j * w3.size() + k] += wt;
		}
	}
}

float cost(int target) {
	float sum = 0;
	std::vector<float> target_arr(a3.size());
	target_arr[target] = 1;

	// mean squared error
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

	populateInputs(1);
	forwardProp();
	outputVec(a3);
	std::cout << "cost: " << cost(getLabel(1));
}
