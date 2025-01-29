#include <iostream>
#include <vector>
#include <random>
#include <fstream>

const std::string training_images_filepath = "./train-images.idx3-ubyte";
const std::string training_labels_filepath = "./train-labels.idx1-ubyte";


// random float gen, e.g. rand_real(e2);
std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> rand_real(0, 1); 


//28x28
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

int main()
{
	std::vector<std::vector<unsigned char>> images = readImages(training_images_filepath);
	std::vector<std::vector<unsigned char>> labels = readLabels(training_labels_filepath);

	// display number in text
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			std::cout << (int)images[1][i * 28 + j] << " ";
		}
		std::cout << "\n";
	}
	// display number's label
	std::cout << (int)labels[1][0] << "\n";
}


