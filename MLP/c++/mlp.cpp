#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <boost/algorithm/string.hpp>

#include "mlp.h"

using std::cout; using std::cerr;
using std::endl; using std::string;
using std::ifstream; using std::ostringstream;
using std::istringstream;
using namespace std;


string file_to_string(std::string path) {
    auto ss = ostringstream{};
    ifstream input_file(path);
    if (!input_file.is_open()) {
        cerr << "Could not open the file - '"
             << path << "'" << endl;
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    return ss.str();
}


vecmatrix* string_vecmatrix(std::string file_contents) {
    char delimiter = ',';

    // TODO: can be done better
    istringstream sstream1(file_contents);
    istringstream sstream2(file_contents);
    istringstream sstream3(file_contents);
    string record;
    
    // get dims for matrix
    int sizeCounter = 0;
    while (std::getline(sstream1, record)) { sizeCounter++; };
    int batchSize = 1;
    int colCounter = 0;
    while (std::getline(sstream2, record)) {
        istringstream line(record);
        while (std::getline(line, record, delimiter)) {
            colCounter++;
        };
        break;
    };
    vecmatrix* vecm = vecmatrix_create(sizeCounter, batchSize, colCounter);
    printf("sizeCounter: %i\n", sizeCounter);
    printf("batchSize: %i\n", batchSize);
    printf("ColCounter: %i\n", colCounter);
    
    sizeCounter = 0;
    while (std::getline(sstream3, record)) {
        istringstream line(record);
        colCounter = 0;
        while (std::getline(line, record, delimiter)) {
            vecm->matrix[sizeCounter][batchSize-1][colCounter] = stod(record.c_str());
            colCounter++;
        }
        sizeCounter++;
    }
    return vecm;
}


int read_data(vecmatrix* x, vecmatrix* y, std::string path, bool test) {
    if ( test ) {
        *x = *vecmatrix_create(4, 1, 2);
        *y = *vecmatrix_create(4, 1, 1);
        
        // XOR test data
        x->matrix[0][0][0] = 0;
        x->matrix[0][0][1] = 0;
        x->matrix[1][0][0] = 0;
        x->matrix[1][0][1] = 1;
        x->matrix[2][0][0] = 1;
        x->matrix[2][0][1] = 0;
        x->matrix[3][0][0] = 1;
        x->matrix[3][0][1] = 1;
        
        y->matrix[0][0][0] = 0;
        y->matrix[1][0][0] = 1;
        y->matrix[2][0][0] = 1;
        y->matrix[3][0][0] = 0;
    } else {
        // file content to string
        auto ss_x = file_to_string(path);
        boost::replace_all(path, ".csv", "_y.csv");
        auto ss_y = file_to_string(path);
        
        // read file to vecmatrix
        *x = *string_vecmatrix(ss_x);
        *y = *string_vecmatrix(ss_y);
    }
    return x->ncol;
}


void print_usage() {
    cout << "Usage: " << endl;
    cout << "       mlp <data-path> <layer0> <layer1> <layer2> <learning-rate> <epoch> <seed> <test>" << endl;
}


int main(int argc, char* argv[])
{

    // check inputs
    if ( argc < 9 ) {
        print_usage();
        return -1;
    }
    
    std::string stest(argv[8]);
    bool btest = false;
    if (stest == "true") btest = true;
    
    // read data
    vecmatrix x, y;
    int input_size = read_data(&x, &y, std::string(argv[1]), btest);
    
    printf("######## Data ########\n");
    vecmatrix_print(&x);
    vecmatrix_print(&y);
    printf("Input Size: %i\n", input_size);
    
    // get hyperparameters
    int layers[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    double learning_rate = atof(argv[5]);
    int epochs = atoi(argv[6]);
    
    // build and train network
    nn* mlp = network_create(input_size, atoi(argv[4]), 1, learning_rate);
    network_train_epoch(mlp, &x, &y, epochs);
    
    // test network
    printf("######## Final Pred ########\n");
    std::vector<double> y_pred = network_predict(mlp, &x);
    printf("AUC: %f\n", AUROC(vecmatrix_to_vector(&y), y_pred, y_pred.size()));
    
    // save network
    network_save(mlp, "output");

    return 0;
}
