#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include "mlp.h"

#include <iostream>
using namespace std;


int read_data(vecmatrix* x, vecmatrix* y, std::string path, bool test) {
    if ( test ) {
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
        
        return 2;
    } else {
        return 0;
    }
}


void print_usage() {
    cout << "Usage: " << endl;
    cout << "       mlp <data-path> <layer0> <layer1> <layer2> <learning-rate> <epoch> <seed>" << endl;
}


int main(int argc, char* argv[])
{

    // check inputs
    if ( argc < 8 ) {
        print_usage();
        return -1;
    }
    
    // read data
    vecmatrix* x = vecmatrix_create(4, 1, 2);
    vecmatrix* y = vecmatrix_create(4, 1, 1);
    int input_size = read_data(x, y, std::string(argv[1]), true);
    
    printf("######## Data ########\n");
    vecmatrix_print(x, true);
    vecmatrix_print(y, true);
    
    // get hyperparameters
    int layers[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    double learning_rate = atof(argv[5]);
    int epochs = atoi(argv[6]);
    
    // build and train network
    nn* mlp = network_create(input_size, atoi(argv[4]), 1, learning_rate);
    network_train_epoch(mlp, x, y, epochs);
    
    // test network
    printf("######## Final Pred ########\n");
    std::vector<double> y_pred = network_predict(mlp, x);
    vector_print(y_pred);
    vector_print(vecmatrix_to_vector(y));

    return 0;
}
