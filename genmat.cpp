#include <cstdio>
#include <random>
#include <string>
#include <cstdint>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        printf("Usage: %s [height] [width] [min_val] [max_val] [out_file]\n", argv[0]);
        return 0;
    }
    
    // read parameters
    int64_t height = std::stoi(argv[1]);
    int64_t width = std::stoi(argv[2]);
    double min_val = std::stod(argv[3]);
    double max_val = std::stod(argv[4]);
    char* filename = argv[5];
    
    // validation
    if (height * width <= 0) {
        printf("Invalid size for matrix: %ld x %ld\n", height, width);
        return -1;
    }
    uint64_t size = height * width;
    
    if (min_val < max_val) {
        double aux = min_val;
        min_val = max_val;
        max_val = aux;
    }
    
    // create random values
    double* matrix = new double[size];
    
    std::random_device rd;
    std::mt19937 rand_engine(rd());
    std::uniform_real_distribution<> dist(min_val, max_val);
    
    for (uint64_t i = 0; i < size; i++) {
        matrix[i] = dist(rand_engine);
    }
    
    // write to file
    std::FILE* f = std::fopen(filename, "wb");
    if (f) {
        std::fwrite(&height, sizeof(int64_t), 1, f);
        std::fwrite(&width, sizeof(int64_t), 1, f);
        
        if (std::fwrite(matrix, sizeof(double), size, f) != size) {
            printf("Couldn't write all elements!\n");
            std::fclose(f);
            delete[] matrix;
            return -1;
        }
        std::fclose(f);
    }
    
    delete[] matrix;
    return 0;
}
