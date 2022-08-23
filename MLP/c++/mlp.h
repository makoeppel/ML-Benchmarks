// code taken and changed from https://github.com/markkraay/mnist-from-scratch


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 100

typedef struct {
    double** m;
    int nrow;
    int ncol;
} matrix;


typedef struct {
    double*** matrix;
    int size;
    int nrow;
    int ncol;
} vecmatrix;


typedef struct {
    int input;
    int hidden;
    int output;
    double learning_rate;
    matrix* hidden_weights;
    matrix* output_weights;
} nn;


void vector_print(std::vector<double> input) {
    for (int i = 0; i < input.size(); i++) {
        printf("%f ", input.at(i));
    }
    printf("\n");
}


std::vector<double> matrix_to_vector(matrix *m) {
    std::vector<double> vec;
    for ( int i = 0; i < m->nrow; i++ ) {
        vec.push_back(m->m[i][0]);
    }
    return vec;
}


std::vector<double> vecmatrix_to_vector(vecmatrix *m) {
    std::vector<double> vec;
    for ( int i = 0; i < m->size; i++ ) {
        for ( int j = 0; j < m->nrow; j++ ) {
            vec.push_back(m->matrix[i][j][0]);
        }
    }
    return vec;
}


matrix* matrix_create(int row, int col) {
    matrix *m = (matrix*) malloc(sizeof(matrix));
    m->nrow = row;
    m->ncol = col;
    m->m = (double**) malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        m->m[i] = (double*) malloc(col * sizeof(double));
    }
    return m;
}


vecmatrix* vecmatrix_create(int size, int row, int col) {
    vecmatrix *vecm = (vecmatrix*) malloc(sizeof(vecmatrix));
    vecm->size = size;
    vecm->nrow = row;
    vecm->ncol = col;
    vecm->matrix = (double***) malloc(size * sizeof(double**));
    for ( int s = 0; s < size; s++ ) {
        vecm->matrix[s] = (double**) malloc(row * sizeof(double*));
        for ( int i = 0; i < row; i++ ) {
            vecm->matrix[s][i] = (double*) malloc(col * sizeof(double));
        }
    }
    return vecm;
}


void matrix_fill(matrix *m, int n) {
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            m->m[i][j] = n;
        }
    }
}


void matrix_free(matrix *m) {
    for (int i = 0; i < m->nrow; i++) {
        free(m->m[i]);
    }
    free(m);
    m = NULL;
}


void matrix_print(matrix *m, bool printAll) {
    printf("(%d, %d)\n", m->nrow, m->ncol);
    if ( printAll ) {
        for (int i = 0; i < m->nrow; i++) {
            for (int j = 0; j < m->ncol; j++) {
                printf("%1.3f ", m->m[i][j]);
            }
            printf("\n");
        }
    }
}


void matrix_print(matrix *m) {
    matrix_print(m, false);
}


void vecmatrix_print(vecmatrix *vecm, bool printAll) {
    printf("(%d, %d, %d)\n", vecm->size, vecm->nrow, vecm->ncol);
    if ( printAll ) {
        for (int i = 0; i < vecm->size; i++) {
            for (int j = 0; j < vecm->nrow; j++) {
                for (int k = 0; k < vecm->ncol; k++) {
                    printf("%1.3f ", vecm->matrix[i][j][k]);
                }
                printf("\n");
            }
        }
    }
}


void vecmatrix_print(vecmatrix *vecm) {
    vecmatrix_print(vecm, false);
}


matrix* matrix_copy(matrix *m) {
    matrix* mat = matrix_create(m->nrow, m->ncol);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            mat->m[i][j] = m->m[i][j];
        }
    }
    return mat;    
}


void matrix_save(matrix* m, char* file_string) {
    FILE* file = fopen(file_string, "w");
    fprintf(file, "%d\n", m->nrow);
    fprintf(file, "%d\n", m->ncol);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            fprintf(file, "%.6f\n", m->m[i][j]);
        }
    }
    printf("Successfully saved matrix to %s\n", file_string);
    fclose(file);
}


matrix* matrix_load(char* file_string) {
    FILE* file = fopen(file_string, "r");
    char entry[MAXCHAR]; 
    fgets(entry, MAXCHAR, file);
    int rows = atoi(entry);
    fgets(entry, MAXCHAR, file);
    int cols = atoi(entry);
    matrix* m = matrix_create(rows, cols);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            fgets(entry, MAXCHAR, file);
            m->m[i][j] = strtod(entry, NULL);
        }
    }
    printf("Sucessfully loaded matrix from %s\n", file_string);
    fclose(file);
    return m;
}


double uniform_distribution(double low, double high) {
    double difference = high - low;
    int scale = 10000;
    int scaled_difference = (int)(difference * scale);
    return low + (1.0 * (rand() % scaled_difference) / scale);
}


void matrix_randomize(matrix* m, int n, int seed) {
    std::srand(seed);
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            m->m[i][j] = uniform_distribution(min, max);
        }
    }
}


void matrix_randomize(matrix* m, int n) {
    matrix_randomize(m, n, 42);
}


int matrix_argmax(matrix* m) {
    // Expects a Mx1 matrix
    double max_score = 0;
    int max_idx = 0;
    for (int i = 0; i < m->nrow; i++) {
        if (m->m[i][0] > max_score) {
            max_score = m->m[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}


matrix* matrix_flatten(matrix* m, int axis) {
    // Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
    matrix* mat;
    if (axis == 0) {
        mat = matrix_create(m->nrow * m->ncol, 1);
    } else if (axis == 1) {
        mat = matrix_create(1, m->nrow * m->ncol);
    } else {
        printf("Argument to matrix_flatten must be 0 or 1");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            if (axis == 0) 
                mat->m[i * m->ncol + j][0] = m->m[i][j];
            else if (axis == 1) 
                mat->m[0][i * m->ncol + j] = m->m[i][j];
        }
    }
    return mat;
}


int check_dimensions(matrix *m1, matrix *m2) {
    if (m1->nrow == m2->nrow && m1->ncol == m2->ncol) return 1;
    return 0;
}


matrix* multiply(matrix *m1, matrix *m2) {
    if (check_dimensions(m1, m2)) {
        matrix *m = matrix_create(m1->nrow, m1->ncol);
        for (int i = 0; i < m1->nrow; i++) {
            for (int j = 0; j < m2->ncol; j++) {
                m->m[i][j] = m1->m[i][j] * m2->m[i][j];
            }
        }
        return m;
    } else {
        printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
        exit(1);
    }
}


matrix* add(matrix *m1, matrix *m2) {
    if (check_dimensions(m1, m2)) {
        matrix *m = matrix_create(m1->nrow, m1->ncol);
        for (int i = 0; i < m1->nrow; i++) {
            for (int j = 0; j < m2->ncol; j++) {
                m->m[i][j] = m1->m[i][j] + m2->m[i][j];
            }
        }
        return m;
    } else {
        printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
        exit(1);
    }
}


matrix* subtract(matrix *m1, matrix *m2) {
    if (check_dimensions(m1, m2)) {
        matrix *m = matrix_create(m1->nrow, m1->ncol);
        for (int i = 0; i < m1->nrow; i++) {
            for (int j = 0; j < m2->ncol; j++) {
                m->m[i][j] = m1->m[i][j] - m2->m[i][j];
            }
        }
        return m;
    } else {
        printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
        exit(1);
    }
}


matrix* apply(double (*func)(double), matrix* m) {
    matrix *mat = matrix_copy(m);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            mat->m[i][j] = (*func)(m->m[i][j]);
        }
    }
    return mat;
}


matrix* dot(matrix *m1, matrix *m2) {
    if (m1->ncol == m2->nrow) {
        matrix *m = matrix_create(m1->nrow, m2->ncol);
        for (int i = 0; i < m1->nrow; i++) {
            for (int j = 0; j < m2->ncol; j++) {
                double sum = 0;
                for (int k = 0; k < m2->nrow; k++) {
                    sum += m1->m[i][k] * m2->m[k][j];
                }
                m->m[i][j] = sum;
            }
        }
        return m;
    } else {
        printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
        exit(1);
    }
}


matrix* scale(double n, matrix* m) {
    matrix* mat = matrix_copy(m);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            mat->m[i][j] *= n;
        }
    }
    return mat;
}


matrix* addScalar(double n, matrix* m) {
    matrix* mat = matrix_copy(m);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            mat->m[i][j] += n;
        }
    }
    return mat;
}


matrix* transpose(matrix* m) {
    matrix* mat = matrix_create(m->ncol, m->nrow);
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            mat->m[j][i] = m->m[i][j];
        }
    }
    return mat;
}


double sigmoid(double input) {
    return 1.0 / (1 + exp(-1 * input));
}


matrix* sigmoidPrime(matrix* m) {
    matrix* ones = matrix_create(m->nrow, m->ncol);
    matrix_fill(ones, 1);
    matrix* subtracted = subtract(ones, m);
    matrix* multiplied = multiply(m, subtracted);
    matrix_free(ones);
    matrix_free(subtracted);
    return multiplied;
}


matrix* softmax(matrix* m) {
    double total = 0;
    for (int i = 0; i < m->nrow; i++) {
        for (int j = 0; j < m->ncol; j++) {
            total += exp(m->m[i][j]);
        }
    }
    matrix* mat = matrix_create(m->nrow, m->ncol);
    for (int i = 0; i < mat->nrow; i++) {
        for (int j = 0; j < mat->ncol; j++) {
            mat->m[i][j] = exp(m->m[i][j]) / total;
        }
    }
    return mat;
}


nn* network_create(int input, int hidden, int output, double lr) {
    nn* net = (nn *) malloc(sizeof(nn));
    net->input = input;
    net->hidden = hidden;
    net->output = output;
    net->learning_rate = lr;
    matrix* hidden_layer = matrix_create(hidden, input);
    matrix* output_layer = matrix_create(output, hidden);
    matrix_randomize(hidden_layer, hidden);
    matrix_randomize(output_layer, output);
    net->hidden_weights = hidden_layer;
    net->output_weights = output_layer;
    return net;
}


std::vector<double> network_predict(nn* net, vecmatrix* x) {
    std::vector<double> pred;
    for ( int j = 0; j < x->size; j++ ) {
        // TODO: can be done better
        matrix* input = matrix_create(x->nrow, x->ncol);
        input->m = x->matrix[j];
        matrix* hidden_inputs   = dot(net->hidden_weights, transpose(input));
        matrix* hidden_outputs  = apply(sigmoid, hidden_inputs);
        matrix* final_inputs    = dot(net->output_weights, hidden_outputs);
        matrix* result          = apply(sigmoid, final_inputs);
        pred.push_back(result->m[0][0]);
    }
    return pred;
}


double network_train(nn* net, matrix* input, matrix* output) {
    // forward
    matrix* hidden_inputs   = dot(net->hidden_weights, transpose(input));
    matrix* hidden_outputs  = apply(sigmoid, hidden_inputs);
    matrix* final_inputs    = dot(net->output_weights, hidden_outputs);
    matrix* final_outputs   = apply(sigmoid, final_inputs);
    
    // errors
    matrix* output_errors = subtract(output, final_outputs);
    matrix* hidden_errors = dot(transpose(net->output_weights), output_errors);
    double error = output_errors->m[0][0];
    
    // backwards
    matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
    matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
    matrix* transposed_mat = transpose(hidden_outputs);
    matrix* dot_mat = dot(multiplied_mat, transposed_mat);
    matrix* scaled_mat = scale(net->learning_rate, dot_mat);
    matrix* added_mat = add(net->output_weights, scaled_mat);
    
    // Free the old weights before replacing
    matrix_free(net->output_weights);
    net->output_weights = added_mat;

    matrix_free(sigmoid_primed_mat);
    matrix_free(multiplied_mat);
    matrix_free(transposed_mat);
    matrix_free(dot_mat);
    matrix_free(scaled_mat);
    
    // Reusing variables after freeing memory
    sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
    multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
    transposed_mat = transpose(input);
    dot_mat = dot(multiplied_mat, input);
    scaled_mat = scale(net->learning_rate, dot_mat);
    added_mat = add(net->hidden_weights, scaled_mat);
    
    // Free the old hidden_weights before replacement
    matrix_free(net->hidden_weights); 
    net->hidden_weights = added_mat;

    matrix_free(sigmoid_primed_mat);
    matrix_free(multiplied_mat);
    matrix_free(transposed_mat);
    matrix_free(dot_mat);
    matrix_free(scaled_mat);

    // Free matrices
    matrix_free(hidden_inputs);
    matrix_free(hidden_outputs);
    matrix_free(final_inputs);
    matrix_free(final_outputs);
    matrix_free(output_errors);
    matrix_free(hidden_errors);
    
    return error;
}


void network_train_epoch(nn* net, vecmatrix* x, vecmatrix* y, int epochs) {
    printf("######## Training ########\n");
    for (int i = 0; i < epochs; i++) {
        double errors = 0;
        for ( int j = 0; j < x->size; j++ ) {
            // TODO: can be done better
            matrix* input = matrix_create(x->nrow, x->ncol);
            matrix* output = matrix_create(y->nrow, y->ncol);
            input->m = x->matrix[j];
            output->m = y->matrix[j];
            errors += pow(network_train(net, input, output), 2);
        }
        if ( i % 100 == 0)
            printf("Epoch: %i, Error: %f\n", i, errors/x->size);
    }
}

