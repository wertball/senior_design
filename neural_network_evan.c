#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#define INPUTS 5
#define H_LAYERS 3
#define H_HEIGHT 20
#define OUTPUTS 1
#define BIAS 1

#define DATA_MIN 0
#define DATA_MAX 300
#define DATA_RANGE (DATA_MAX - DATA_MIN)
#define DATA_FILE "data_for_training.txt"


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
double ALPHA = .2;

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
static inline double normalize(double input) {
    return (input - DATA_MIN) / (DATA_RANGE);
}

static inline double denormalize(double input) {
    return (DATA_RANGE * input) + DATA_MIN;
}
//--------------------------------------------

void forward(double in[INPUTS]) {
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
        //outputs[i] = normalize(outputs[i]);

        //Activation
        outputs[i] = activation(outputs[i]);
        //printf("Output %d: %f\n", i, outputs[i]);
    }
    //-----------------------------------
}

void backpropagation(double in[INPUTS], double target_out[OUTPUTS]) {
    int i,j,k;
    double delta_sum;
    double deltas_h[H_HEIGHT];
    double deltas_h_new[H_HEIGHT];
    double deltas_o[OUTPUTS];
    double weights_out_new[OUTPUTS * H_HEIGHT];
    #if (H_LAYERS > 1)
    //TODO: Make this more memory efficient
    double weights_h_new[H_LAYERS - 1][H_HEIGHT * H_HEIGHT];
    #endif


    //Output layer------------------------------
    for (i = 0; i < OUTPUTS; i++) {
        deltas_o[i] = (outputs[i] * (1 - outputs[i])) * (outputs[i] - target_out[i]);
        for (j = 0; j < H_HEIGHT; j++) {
            weights_out_new[(i * H_HEIGHT) + j] = weights_out[(i * H_HEIGHT) + j] -
                    (ALPHA * deltas_o[i] * h_out[H_LAYERS - 1][j]);
            //printf("New weight_out[%d]: %f\n", (i * OUTPUTS) + j, weights_out_new[(i * OUTPUTS) + j]);
        }
        #if (BIAS > 0)
        weights_bias_out[i] -= ALPHA * deltas_o[i] * BIAS;
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
            weights_h_new[H_LAYERS-2][(i * H_HEIGHT) + j] = weights_h[H_LAYERS-2][(i * H_HEIGHT) + j] -
                    (ALPHA * deltas_h[i] * h_out[H_LAYERS-2][j]);
        }
        #if (BIAS > 0)
        weights_bias_h[H_LAYERS-1][i] -= ALPHA * deltas_h[i] * BIAS;
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
                weights_h_new[i-1][(j * H_HEIGHT) + k] = weights_h[i-1][(j * H_HEIGHT) + k] -
                        (ALPHA * deltas_h_new[j] * h_out[i - 1][k]);
            }
            #if (BIAS > 0)
            weights_bias_h[i][j] -= ALPHA * deltas_h_new[j] * BIAS;
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
            weights_in[(i * INPUTS) + j] -= (ALPHA * deltas_h_new[i] * in[j]);
        }
        #if (BIAS > 0)
        weights_bias_h[0][i] -= ALPHA * deltas_h_new[i] * BIAS;
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
            weights_in[(i * INPUTS) + j] -= (ALPHA * deltas_h[i] * in[j]);
            //printf("New weight_in[%d]: %f\n", (i * INPUTS) + j, weights_in[(i * INPUTS) + j]);
        }
        #if (BIAS > 0)
        weights_bias_h[0][i] -= ALPHA * deltas_h[i] * BIAS;
        #endif
    }
    #endif
    //---------------------------------------------------------------------------------------------

    //Write new values---------------------------
    //Output layer
    memcpy(weights_out, weights_out_new, OUTPUTS * H_HEIGHT * sizeof(double));
//    for (i = 0; i < OUTPUTS * H_HEIGHT; i++) {
//        weights_out[i] = weights_out_new[i];
//    }
    //Hidden layer
    #if (H_LAYERS > 1)
    for (i = 0; i < H_LAYERS - 1; i++) {
        memcpy(weights_h, weights_h_new, H_HEIGHT * H_HEIGHT * sizeof(double));
//        for (j = 0; j < (H_HEIGHT * H_HEIGHT); j++) {
//            weights_h[i][j] = weights_h_new[i][j];
//        }
    }
    #endif
    //--------------------------------------------
}



int read_training_set(char *file_name, double ***training_in, double **training_out) {
    int i, j, samples, current_line;
    double **t_in, *t_out;
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

    //Allocate training arrays
    t_in = (double **) malloc(samples * sizeof(double *));
    for (i = 0; i < samples; i++) {
        t_in[i] = (double *) malloc(INPUTS * sizeof(double));
    }
    t_out = (double *) malloc(samples * sizeof(double));
    *training_in = t_in;
    *training_out = t_out;

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
                t_in[i][j] = normalize(t_in[i][j]);
                pointer = strtok(NULL, "\t");
            }
            if (pointer == NULL) {
                printf("Error: Unexpected end of line: %d\n", current_line);
                fflush(stdout);
                return -1;
            }
            sscanf(pointer, "%lf", &t_out[i]);
            t_out[i] = normalize(t_out[i]);
            i++;
        }
        current_line++;
    }
    //close file
    fclose(fp);
    return samples;
}

#define SAMPLES 10
int main() {
    int i, j, k;
    int samples;
    unsigned long int it = 0;
    struct timeval t1, t2;
    double **training_in, *training_out;
    double error;

    //Z = X^2 + Y^2
    //double training_in[SAMPLES][2], training_out[SAMPLES];
    //for (i = 0; i < SAMPLES; i++) {
    //    training_in[i][0] = normalize(11-i);
    //    training_in[i][1] = normalize(i);
    //    training_out[i] = normalize(((11-i) * (11-i)) + (i * i));
    //}


    samples = read_training_set(DATA_FILE, &training_in, &training_out);
    if (samples <= 0) {
        return 1;
    }
    initialize();


    gettimeofday(&t1, NULL);
    error = 1;
    while (error > 1E-10) {
        error = 0;

        //Forward and back for each sample
        for (i = 0; i < samples; i++) {
            forward(training_in[i]);
            error += (training_out[i] - outputs[0]) * (training_out[i] - outputs[0]);
            backpropagation(training_in[i], &training_out[i]);
        }
        error /= samples;

        //Debug prints
        if ((it & 0xFFFF) == 0) {
            for (j = 0; j < samples; j++) {
                forward(training_in[j]);
                printf("%2.0f %2.0f %3.0f %3.0f %4.2f %9.7f\n",
                       denormalize(training_in[j][0]),
                       denormalize(training_in[j][1]),
                       denormalize(training_in[j][2]),
                       denormalize(training_in[j][3]),
                       denormalize(training_in[j][4]),
                       denormalize(outputs[0]));
            }
            printf("Iteration: %lu\n", it);
            printf("Total Error: %.3e\n", error);
            fflush(stdout);
        }
        it++;
        //printf("Iteration:%lu  Error:%12.10f\n", it++, error);
    }

    //while (error > 1E-6) {
    //    error = 0;
    //    for (i = 0; i < SAMPLES; i++) {
    //        forward(training_in[i]);
    //        error += (training_out[i] - outputs[0]) * (training_out[i] - outputs[0]);
    //        backpropagation(training_in[i], &training_out[i]);
    //    }
    //    error /= SAMPLES;
    //    if ((it & 0xFFFF) == 0) {
    //        for (j = 0; j < SAMPLES; j++) {
    //            forward(training_in[j]);
    //            for (k = 0; k < INPUTS; k++) {
    //                printf("%f ", denormalize(training_in[j][k]));
    //            }
    //            printf("%f\n", denormalize(outputs[0]));
    //        }
    //        printf("Iteration: %lu\n", it);
    //        printf("Total Error: %.3e\n", error);
    //        fflush(stdout);
    //    }
    //    it++;
    //}
    gettimeofday(&t2, NULL);
    printf("Training time: %f\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-6);
    printf("Iterations: %lu\n", it);

    error = 0;
    for (i = 0; i < samples; i++) {
        forward(training_in[i]);
        error += (training_out[i] - outputs[0]) * (training_out[i] - outputs[0]);
        printf("%2.0f %2.0f %3.0f %3.0f %4.2f %9.7f\n",
               denormalize(training_in[i][0]),
               denormalize(training_in[i][1]),
               denormalize(training_in[i][2]),
               denormalize(training_in[i][3]),
               denormalize(training_in[i][4]),
               denormalize(outputs[0]));
    }
    printf("Total Error: %.3e\n", error / samples);

    //LET MY ARRAYS GO
    for (i = 0; i < samples; i++) {
        free(training_in[i]);
    }
    free(training_in);
    free(training_out);
    return 0;
}
