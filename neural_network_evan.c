#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <float.h>

#define INPUTS 5
#define H_LAYERS 3
#define H_HEIGHT 10
#define OUTPUTS 1
#define BIAS 1

#define DATA_FILE "data_for_training.txt"
#define TEST_FILE "data_for_verify.txt"

#define ITERATIONS 1E4

//#define DEBUG

//Weight declarations
double weights_in[INPUTS * H_HEIGHT];// = {.15,.20};
double weights_out[OUTPUTS * H_HEIGHT];// = {.65,.7};
#if (BIAS > 0)
double weights_bias_out[OUTPUTS];// = {0.85};
double weights_bias_h[H_LAYERS][H_HEIGHT];// = {{.3,.35}, {.6,.75}};
#endif
#if (H_LAYERS > 1)
double weights_h[H_LAYERS - 1][H_HEIGHT * H_HEIGHT];// = {.4,.45,.5,.55};
#endif

//I/O declarations
double h_out[H_LAYERS][H_HEIGHT];
double outputs[OUTPUTS];

//Learning declarations
double ALPHA = .5;

//Randomization functions----------------------
void randomize_array(int length, double arr[length]) {
    int i;
    for (i = 0; i < length; i++) {
//        arr[i] = (rand() & 0b11) + 1;
        arr[i] = 1.0f / ((rand() & 0xF) + 1);
        if ((rand() & 1) == 0) {
            arr[i] *= -1;
        }
    }
}

void randomize_2d_array(int rows, int columns, double arr[rows][columns]) {
    int i;
    for (i = 0; i < rows; i++) {
        randomize_array(columns, arr[i]);
    }
}
//------------------------------------------------

//Initialization of variables
static inline void initialize() {
    //Randomize weights
    srand(time(NULL));
    randomize_array(INPUTS * H_HEIGHT, weights_in);
    randomize_array(OUTPUTS * H_HEIGHT, weights_out);
    #if (BIAS > 0)
    randomize_array(OUTPUTS, weights_bias_out);
    randomize_2d_array(H_LAYERS, H_HEIGHT, weights_bias_h);
    #endif
    #if (H_LAYERS > 1)
    randomize_2d_array(H_LAYERS - 1, H_HEIGHT * H_HEIGHT, weights_h);
    #endif
}

//Sigmoid activation function
static inline double activation(double input) {
    return 1.0f / (1.0f + exp(-1 * input));
}

//Normalization functions--------------------
static inline double normalize(double input, double min, double max) {
    return (input - min) / (max - min);
}

static inline double denormalize(double input, double min, double max) {
    return ((max - min) * input) + min;
}
//--------------------------------------------

void forward(double in[INPUTS], double *output_range) {
    int i,j,k;

    //Input layer-----------------------
    for (i = 0; i < H_HEIGHT; i++) {
        //Input weighted sum
        h_out[0][i] = 0;
        for (j = 0; j < INPUTS; j++) {
            h_out[0][i] += weights_in[(i * INPUTS) + j] * in[j];
        }

        //Bias weighted sum
        #if (BIAS > 0)
        h_out[0][i] += weights_bias_h[0][i] * BIAS;
        #endif

        //Activation
        h_out[0][i] = activation(h_out[0][i]);
        //printf("h_out[0][%d]: %f\n", i, h_out[0][i]);
    }
    //------------------------------------

    //Hidden Layers----------------------------
    #if (H_LAYERS > 1)
    for (i = 0; i < (H_LAYERS - 1); i++) { //For each layer
        for (j = 0; j < H_HEIGHT; j++) { //For each node in a layer
            //Hidden weighted sum
            h_out[i+1][j] = 0;
            for (k = 0; k < H_HEIGHT; k++) { //For each node in previous layer
                h_out[i+1][j] += weights_h[i][(j * H_HEIGHT) + k] * h_out[i][k];
            }

            //Bias weighted sum
            #if (BIAS > 0)
            h_out[i+1][j] += weights_bias_h[i+1][j] * BIAS;
            #endif

            //Activation
            h_out[i+1][j] = activation(h_out[i+1][j]);
            //printf("h_out[%d][%d]: %f\n", i+1, j, h_out[i+1][j]);
        }
    }
    #endif
    //-----------------------------------------

    //Output Layer---------------------
    for (i = 0; i < OUTPUTS; i++) {
        //Hidden layer weighted sum
        outputs[i] = 0;
        for (j = 0; j < H_HEIGHT; j++) {
            outputs[i] += weights_out[(i * H_HEIGHT) + j] * h_out[H_LAYERS - 1][j];
        }

        //Bias weighted sum
        #if (BIAS > 0)
        outputs[i] += weights_bias_out[i] * BIAS;
        #endif

        //Normalize output
        outputs[i] = normalize(outputs[i], output_range[0], output_range[1]);

        //Activation
        //outputs[i] = activation(outputs[i]);
        //printf("Output %d: %f\n", i, outputs[i]);
    }
    //-----------------------------------
}

void backpropagation(double in[INPUTS], double target_out[OUTPUTS],
                        double weights_in_delta[INPUTS * H_HEIGHT], double weights_out_delta[OUTPUTS * H_HEIGHT]
                        #if (BIAS > 0)
                        ,double weights_bias_out_delta[OUTPUTS], double weights_bias_h_delta[H_LAYERS][H_HEIGHT]
                        #endif
                        #if (H_LAYERS > 1)
                        ,double weights_h_delta[H_LAYERS-1][H_HEIGHT*H_HEIGHT]
                        #endif
                        ) {
    int i,j,k;
    double delta_sum;
    double deltas_h[H_HEIGHT];
    double deltas_h_new[H_HEIGHT];
    double deltas_o[OUTPUTS];

    //Output layer------------------------------
    for (i = 0; i < OUTPUTS; i++) {
        deltas_o[i] = (outputs[i] - target_out[i]);
        for (j = 0; j < H_HEIGHT; j++) {
            weights_out_delta[(i * H_HEIGHT) + j] += deltas_o[i] * h_out[H_LAYERS - 1][j];
        }
        #if (BIAS > 0)
        weights_bias_out_delta[i] += deltas_o[i] * BIAS;
        #endif
    }
    //-------------------------------------------

    //Hidden layers--------------------------------------------------------------------------------
    #if (H_LAYERS > 1)
    //Layer connected to outputs
    for (i = 0; i < H_HEIGHT; i++) { //For each node in hidden layer
        //Get delta---------
        delta_sum = 0;
        for (j = 0; j < OUTPUTS; j++) {  //For each output
            delta_sum += weights_out[i + (j * H_HEIGHT)] * deltas_o[j];
        }
        deltas_h[i] = (h_out[H_LAYERS-1][i] * (1 - h_out[H_LAYERS-1][i])) * delta_sum;
        //------------------

        //Calculate new weights------------
        for (j = 0; j < H_HEIGHT; j++) {
            weights_h_delta[H_LAYERS-2][(i * H_HEIGHT) + j] += deltas_h[i] * h_out[H_LAYERS-2][j];
        }
        #if (BIAS > 0)
        weights_bias_h_delta[H_LAYERS-1][i] += deltas_h[i] * BIAS;
        #endif
        //----------------------------------
    }
    //Each hidden layer not connected to input/hidden output layer
    for (i = H_LAYERS-2; i > 0; i--) {
        for (j = 0; j < H_HEIGHT; j++) {
            //Get delta------
            delta_sum = 0;
            for (k = 0; k < H_HEIGHT; k++) {
                delta_sum += weights_h[i][j + (k * H_HEIGHT)] * deltas_h[k];
            }
            deltas_h_new[j] = (h_out[i][j] * (1 - h_out[i][j])) * delta_sum;
            //---------------

            //Get new weights-------------------
            for (k = 0; k < H_HEIGHT; k++) {
                weights_h_delta[i-1][(j * H_HEIGHT) + k] += (deltas_h_new[j] * h_out[i - 1][k]);
            }
            #if (BIAS > 0)
            weights_bias_h_delta[i][j] += deltas_h_new[j] * BIAS;
            #endif
            //----------------------------------
        }
        memcpy(deltas_h, deltas_h_new, H_HEIGHT * sizeof(double));
    }
    //Layer connected to inputs
    for (i = 0; i < H_HEIGHT; i++) {
        //Get delta-----------------
        delta_sum = 0;
        for (j = 0; j < H_HEIGHT; j++) {
            delta_sum += weights_h[0][i + (j * H_HEIGHT)] * deltas_h[j];
        }
        deltas_h_new[i] = (h_out[0][i] * (1 - h_out[0][i])) * delta_sum;
        //--------------------------

        //Get new weights---------------
        for (j = 0; j < INPUTS; j++) {
            weights_in_delta[(i * INPUTS) + j] += (deltas_h_new[i] * in[j]);
        }
        #if (BIAS > 0)
        weights_bias_h_delta[0][i] += deltas_h_new[i] * BIAS;
        #endif
        //-------------------------------
    }
    #else
    for (i = 0; i < H_HEIGHT; i++) {
        delta_sum = 0;
        for (j = 0; j < OUTPUTS; j++) {
            delta_sum += weights_out[i + (j * OUTPUTS)] * deltas_o[j];
        }
        deltas_h[i] = (h_out[0][i] * (1 - h_out[0][i])) * delta_sum;
        for (j = 0; j < INPUTS; j++) {
            weights_in_delta[(i * INPUTS) + j] += (deltas_h[i] * in[j]);
            //printf("New weight_in[%d]: %f\n", (i * INPUTS) + j, weights_in[(i * INPUTS) + j]);
        }
        #if (BIAS > 0)
        weights_bias_h_delta[0][i] += deltas_h[i] * BIAS;
        #endif
    }
    #endif
    //---------------------------------------------------------------------------------------------

}

int load_file(char *file_name, double ***input_samples, double **output_samples, double ***data_range) {
    int i, j, samples, current_line;
    double **t_in, *t_out, **d_range;
    char temp_str[100];
    char *pointer;
    FILE *fp;

    //Open file
    fp = fopen(file_name, "r");
    if (!fp) {
        printf("Error: Failed to open %s\n", file_name);
        fflush(stdout);
        return -1;
    }

    //count number of samples
    samples = 0;
    while (fgets(temp_str, 100, fp) != NULL) {
        if (strlen(temp_str) > 2) {
            samples++;
        }
    }
    if (samples == 0) {
        printf("Error: No data in %s\n", file_name);
        fflush(stdout);
        return -1;
    }
    rewind(fp);

    //Allocate arrays
    t_in = (double **) malloc(samples * sizeof(double *));
    d_range = (double **) malloc((INPUTS + 1) * sizeof(double *));
    for (i = 0; i < samples; i++) {
        t_in[i] = (double *) malloc(INPUTS * sizeof(double));
    }
    for (i = 0; i < (INPUTS + 1); i++) {
        d_range[i] = (double *) malloc(2 * sizeof(double));
        d_range[i][0] = DBL_MAX;
        d_range[i][1] = DBL_MIN;
    }
    t_out = (double *) malloc(samples * sizeof(double));
    *input_samples = t_in;
    *output_samples = t_out;
    *data_range = d_range;

    //Get data from file
    i = 0;
    current_line = 1;
    while (fgets(temp_str, 100, fp) != NULL) {
        if (strlen(temp_str) > 2) {
            pointer = strtok(temp_str, "\t");
            for (j = 0; j < INPUTS; j++) {
                if (pointer == NULL) {
                    printf("Error: Unexpected end of line: %d\n", current_line);
                    fflush(stdout);
                    return -1;
                }
                sscanf(pointer, "%lf", &t_in[i][j]);
                if (t_in[i][j] < d_range[j][0]) {
                    d_range[j][0] = t_in[i][j];
                }
                if (t_in[i][j] > d_range[j][1]) {
                    d_range[j][1] = t_in[i][j];
                }
                pointer = strtok(NULL, "\t");
            }
            if (pointer == NULL) {
                printf("Error: Unexpected end of line: %d\n", current_line);
                fflush(stdout);
                return -1;
            }
            sscanf(pointer, "%lf", &t_out[i]);
            if (t_out[i] < d_range[INPUTS][0]) {
                d_range[INPUTS][0] = t_out[i];
            }
            if (t_out[i] > d_range[INPUTS][1]) {
                d_range[INPUTS][1] = t_out[i];
            }
            i++;
        }
        current_line++;
    }
    //close file
    fclose(fp);

    return samples;
}

void normalize_data(double **data_in, double *data_out, double **data_range, int samples) {
    int i, j;

    //Normalize
    for (i = 0; i < samples; i++) {
        for (j = 0; j < INPUTS; j++) {
            if (data_range[j][0] == data_range[j][1]) {
                data_in[i][j] = 1;
            }
            else {
                data_in[i][j] = normalize(data_in[i][j], data_range[j][0], data_range[j][1]);
            }
        }
        data_out[i] = normalize(data_out[i], data_range[INPUTS][0], data_range[INPUTS][1]);
    }

}

void train(double **training_in, double *training_out, double **data_range, int samples) {
    int i, j;
    double error;
    long long it = 0;
    struct timeval t1, t2;

    double weights_in_delta[INPUTS*H_HEIGHT], weights_out_delta[OUTPUTS*H_HEIGHT];
    #if (BIAS > 0)
    double weights_bias_out_delta[OUTPUTS], weights_bias_h_delta[H_LAYERS][H_HEIGHT];
    #endif
    #if (H_LAYERS>1)
    double weights_h_delta[H_LAYERS-1][H_HEIGHT*H_HEIGHT];
    #endif

    gettimeofday(&t1, NULL);
    error = 1;
    while (it < ITERATIONS) {
        error = 0;

        //Zero delta arrays
        memset(weights_in_delta, 0, INPUTS*H_HEIGHT*sizeof(double));
        memset(weights_out_delta, 0, OUTPUTS*H_HEIGHT*sizeof(double));
        #if (BIAS>0)
        memset(weights_bias_out_delta, 0, OUTPUTS*sizeof(double));
        memset(weights_bias_h_delta, 0, H_LAYERS*H_HEIGHT*sizeof(double));
        #endif
        #if (H_LAYERS>1)
        memset(weights_h_delta, 0, (H_LAYERS-1)*H_HEIGHT*H_HEIGHT*sizeof(double));
        #endif

        //Forward and back for each sample
        for (i = 0; i < samples; i++) {
            forward(training_in[i], data_range[INPUTS]);
            error += fabs(denormalize(outputs[0], data_range[INPUTS][0], data_range[INPUTS][1])
                          - denormalize(training_out[i], data_range[INPUTS][0], data_range[INPUTS][1]))
                          / denormalize(training_out[i], data_range[INPUTS][0], data_range[INPUTS][1]);
            backpropagation(training_in[i], &training_out[i],
                            weights_in_delta,weights_out_delta
            #if (BIAS>0)
                    ,weights_bias_out_delta,weights_bias_h_delta
            #endif
            #if (H_LAYERS>1)
                    ,weights_h_delta
            #endif
            );
        }
        error *= (100.0f / samples);

        //Average weight deltas and apply
        for (i = 0; i < INPUTS*H_HEIGHT; i++) {
            weights_in[i] -= ((ALPHA * weights_in_delta[i]) / samples);
        }
        for (i = 0; i < OUTPUTS * H_HEIGHT; i++) {
            weights_out[i] -= ((ALPHA * weights_out_delta[i]) / samples);
        }
        #if (BIAS>0)
        for (i = 0; i < OUTPUTS; i++) {
            weights_bias_out[i] -= ((ALPHA * weights_bias_out_delta[i]) / samples);
        }
        for (i = 0; i < H_LAYERS; i++) {
            for (j = 0; j < H_LAYERS; j++) {
                weights_bias_h[i][j] -= ((ALPHA * weights_bias_h_delta[i][j]) / samples);
            }
        }
        #endif
        #if (H_LAYERS>1)
        for (i = 0; i < (H_LAYERS-1); i++) {
            for (j = 0; j < (H_HEIGHT*H_HEIGHT); j++) {
                weights_h[i][j] -= ((ALPHA * weights_h_delta[i][j]) / samples);
            }
        }
        #endif


        //Debug prints
        if ((it & 0xFFFF) == 0) {
            #ifdef DEBUG
            for (j = 0; j < samples; j++) {
                forward(training_in[j], data_range[INPUTS]);
                printf("%2.0f %2.0f %3.0f %3.0f %4.2f %9.7f\n",
                       denormalize(training_in[j][0], data_range[0][0], data_range[0][1]),
                       denormalize(training_in[j][1], data_range[1][0], data_range[1][1]),
                       denormalize(training_in[j][2], data_range[2][0], data_range[2][1]),
                       denormalize(training_in[j][3], data_range[3][0], data_range[3][1]),
                       denormalize(training_in[j][4], data_range[4][0], data_range[4][1]),
                       denormalize(outputs[0], data_range[5][0], data_range[5][1]));
            }
            #endif
            printf("Iteration: %.3e\n", (double) it);
            printf("Average %%Error: %.3f%%\n", error);
            fflush(stdout);
        }
        it++;
    }

    gettimeofday(&t2, NULL);
    printf("Training time: %f\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-6);
    printf("Iterations: %llu\n", it);

    error = 0;
    for (i = 0; i < samples; i++) {
        forward(training_in[i], data_range[INPUTS]);
        error += fabs(denormalize(outputs[0], data_range[INPUTS][0], data_range[INPUTS][1])
                      - denormalize(training_out[i], data_range[INPUTS][0], data_range[INPUTS][1]))
                      / denormalize(training_out[i], data_range[INPUTS][0], data_range[INPUTS][1]);
        printf("%2.0f %2.0f %3.0f %3.0f %4.2f %9.7f\n",
               denormalize(training_in[i][0], data_range[0][0], data_range[0][1]),
               denormalize(training_in[i][1], data_range[1][0], data_range[1][1]),
               denormalize(training_in[i][2], data_range[2][0], data_range[2][1]),
               denormalize(training_in[i][3], data_range[3][0], data_range[3][1]),
               denormalize(training_in[i][4], data_range[4][0], data_range[4][1]),
               denormalize(outputs[0], data_range[5][0], data_range[5][1]));
    }
    printf("Average %%Error: %.3f%%\n", error * (100.0f / samples));
}

int test(char *filename, double **data_range) {
    int i;
    int samples;
    double **test_in, *test_out, **test_range;
    double sample_error, average_error, max_error;

    //Load test data and normalize
    samples = load_file(filename, &test_in, &test_out, &test_range);
    normalize_data(test_in, test_out, data_range, samples);
    if (samples <= 0) {
        return 1;
    }

    //Run tests
    average_error = max_error = 0;
    for (i = 0; i < samples; i++) {
        forward(test_in[i], data_range[INPUTS]);

        sample_error = fabs(denormalize(outputs[0], data_range[INPUTS][0], data_range[INPUTS][1]) - denormalize(test_out[i], data_range[INPUTS][0], data_range[INPUTS][1]))
                      / denormalize(test_out[i], data_range[INPUTS][0], data_range[INPUTS][1]);

        average_error += sample_error;
        if (sample_error > max_error) {
            max_error = sample_error;
        }

        printf("%2.0f %2.0f %3.0f %3.0f %4.2f %9.7f    %.3f%%\n",
               denormalize(test_in[i][0], data_range[0][0], data_range[0][1]),
               denormalize(test_in[i][1], data_range[1][0], data_range[1][1]),
               denormalize(test_in[i][2], data_range[2][0], data_range[2][1]),
               denormalize(test_in[i][3], data_range[3][0], data_range[3][1]),
               denormalize(test_in[i][4], data_range[4][0], data_range[4][1]),
               denormalize(outputs[0], data_range[5][0], data_range[5][1]),
               sample_error * 100);
    }
    printf("Average Testing %%Error: %.3f%%\n", average_error * (100.0f / samples));
    printf("Max %%Error: %.3f%%\n", max_error*100);


}

int main() {
    int i;
    int samples;
    double **training_in, *training_out, **data_range;

    samples = load_file(DATA_FILE, &training_in, &training_out, &data_range);
    normalize_data(training_in, training_out, data_range, samples);
    if (samples <= 0) {
        return 1;
    }
    initialize();

    printf("Training...\n");
    train(training_in, training_out, data_range, samples);

    printf("Testing...\n");
    test(TEST_FILE, data_range);

    //Free dynamic arrays
    for (i = 0; i < samples; i++) {
        free(training_in[i]);
    }
    free(training_in);
    for (i = 0; i < (INPUTS + 1); i++) {
        free(data_range[i]);
    }
    free(data_range);
    free(training_out);
    return 0;
}
