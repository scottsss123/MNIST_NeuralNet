#include "includes.h"

#include "NeuralNet.h"

uint32_t reverseInt(uint32_t i) {
	uint8_t c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + ((uint32_t)c4);
}

std::vector<std::vector<uint8_t>> readMNISTImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Could not open file: " << filename << std::endl;
        return {};
    }

    uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    magic_number = reverseInt(magic_number);
    num_images = reverseInt(num_images);
    num_rows = reverseInt(num_rows);
    num_cols = reverseInt(num_cols);

    std::cout << "Number of images in " << filename << " : " << num_images << std::endl;
    std::cout << "Image dimensions: " << num_rows << " x " << num_cols << std::endl;

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_rows * num_cols));
    for (uint32_t i = 0; i < num_images; i++) {
        file.read((char*)images[i].data(), num_rows * num_cols);
    }

    return images;
}

std::vector<uint8_t> readMNISTLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Could not open file: " << filename << std::endl;
        return {};
    }

    uint32_t magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    magic_number = reverseInt(magic_number);
    num_items = reverseInt(num_items);

    std::cout << "Number of labels in " << filename << " : " << num_items << std::endl;

    std::vector<uint8_t> labels(num_items);
    file.read((char*)labels.data(), num_items);

    return labels;
}

int main(int argc, char** argv) {
    // load data
    const std::string training_images_filename = "train-images.idx3-ubyte";
    const std::string training_labels_filename = "train-labels.idx1-ubyte";
    const std::string test_images_filename = "t10k-images.idx3-ubyte";
    const std::string test_labels_filename = "t10k-labels.idx1-ubyte";


	std::vector<uint8_t> training_labels = readMNISTLabels(training_labels_filename);
    std::vector<uint8_t> test_labels = readMNISTLabels(test_labels_filename);
    std::vector<std::vector<uint8_t>> training_images = readMNISTImages(training_images_filename);
    std::vector<std::vector<uint8_t>> test_images = readMNISTImages(test_images_filename);

    // Display a sample image
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (test_images[2][i * 28 + j] > 127 ? '#' : '.') << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << "Label: " << static_cast<int>(test_labels[2]) << std::endl;

    std::vector<int> layerSizes = { 28*28 , 16, 16, 10 };
    NeuralNet network(layerSizes);
    // network.OutputLayerWeights();  // BIGOUTPUT of 728*16 + 16*16 + 16*10  numbers
    // test classification
    std::pair<int, double> guess = network.Classify(testImages[2]);
    std::cout << "guess : " << guess.first << "/nconfidence : " << guess.second << "\n";
}