#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define INPUTS 1
#define H_LAYERS 4
#define H_HEIGHT 5
#define OUTPUTS 1
#define BIAS 1

#define DATA_MIN 0
#define DATA_MAX 256
#define DATA_RANGE (DATA_MAX - DATA_MIN)

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
double ALPHA = 0.5;

//Randomization functions----------------------
void randomize_array(int length, double arr[length]) {
    int i;
    for (i = 0; i < length; i++) {
        arr[i] = ((rand() & 0b11) + 1) / ((rand() & 0xF) + 1);
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
    return (DATA_RANGE) * input + DATA_MIN;
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
            outputs[i] += weights_out[(i * OUTPUTS) + j] * h_out[H_LAYERS - 1][j];
        }

        //Bias weighted sum
        #if (BIAS > 0)
        outputs[i] += weights_bias_out[i] * BIAS;
        #endif

        //Normalize output
        //outputs[i] = normalize(outputs[i]);

        //Activation
        //outputs[i] = activation(outputs[i]);
        //printf("Output %d: %f\n", i, outputs[i]);
    }
    //-----------------------------------
}

void backpropagation(double in[INPUTS], double target_out[OUTPUTS]) {
    int i,j,k;
    double delta_sum;
    double deltas_h[H_HEIGHT];
    double deltas_o[OUTPUTS];
    double weights_out_new[OUTPUTS * H_HEIGHT];
    #if (H_LAYERS > 1)
    //TODO: Make this more memory efficient
    double weights_h_new[H_LAYERS - 1][H_HEIGHT * H_HEIGHT];
    #endif


    //Output layer------------------------------
    for (i = 0; i < OUTPUTS; i++) {
        deltas_o[i] = (outputs[i] - target_out[i]);
        for (j = 0; j < H_HEIGHT; j++) {
            weights_out_new[(i * OUTPUTS) + j] = weights_out[(i * OUTPUTS) + j] -
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
        delta_sum = 0;
        for (j = 0; j < OUTPUTS; j++) {  //For each output
            delta_sum += weights_out[i + (j * H_HEIGHT)] * deltas_o[j];
        }
        deltas_h[i] = (h_out[H_LAYERS-1][i] * (1 - h_out[H_LAYERS-1][i])) * delta_sum;
        for (j = 0; j < H_HEIGHT; j++) {
            weights_h_new[H_LAYERS-2][(i * H_HEIGHT) + j] = weights_h[H_LAYERS-2][(i * H_HEIGHT) + j] -
                    (ALPHA * deltas_h[i] * h_out[H_LAYERS-2][j]);
        }
        #if (BIAS > 0)
        weights_bias_h[H_LAYERS-1][i] -= ALPHA * deltas_h[i] * BIAS;
        #endif
    }
    //Each hidden layer not connected to input/hidden output layer
    for (i = H_LAYERS-2; i > 0; i--) {
        for (j = 0; j < H_HEIGHT; j++) {
            delta_sum = 0;
            for (k = 0; k < H_HEIGHT; k++) {
                delta_sum += weights_h[i][j + (k * H_HEIGHT)] * deltas_h[k];
            }
            deltas_h[j] = (h_out[i][j] * (1 - h_out[i][j])) * delta_sum;
            for (k = 0; k < H_HEIGHT; k++) {
                weights_h_new[i-1][(j * H_HEIGHT) + k] = weights_h[i-1][(j * H_HEIGHT) + k] -
                        (ALPHA * deltas_h[j] * h_out[i - 1][k]);
            }
            #if (BIAS > 0)
            weights_bias_h[i][j] -= ALPHA * deltas_h[j] * BIAS;
            #endif
        }
    }
    //Layer connected to inputs
    for (i = 0; i < H_HEIGHT; i++) {
        delta_sum = 0;
        for (j = 0; j < H_HEIGHT; j++) {
            delta_sum += weights_h[0][i + (j * H_HEIGHT)] * deltas_h[j];
        }
        deltas_h[i] = (h_out[0][i] * (1 - h_out[0][i])) * delta_sum;
        for (j = 0; j < INPUTS; j++) {
            weights_in[(i * INPUTS) + j] -= (ALPHA * deltas_h[i] * in[j]);
        }
        #if (BIAS > 0)
        weights_bias_h[0][i] -= ALPHA * deltas_h[i] * BIAS;
        #endif
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
    for (i = 0; i < OUTPUTS * H_HEIGHT; i++) {
        weights_out[i] = weights_out_new[i];
    }
    //Hidden layer
    #if (H_LAYERS > 1)
    for (int i = 0; i < H_LAYERS - 1; i++) {
        for (int j = 0; j < (H_HEIGHT * H_HEIGHT); j++) {
            weights_h[i][j] = weights_h_new[i][j];
        }
    }
    #endif
    //--------------------------------------------
}

/*void print_weights() {
    int i, j;
    printf("input weights:\n")
    for (i = 0; i < INPUTS * H_HEIGHT; i++) {
        printf("%f  ", weights_in[i]);
    }

}*/

int main() {
    int i,j;
    unsigned long int it = 0;
    struct timeval t1, t2;
//    double training_in[] = {0.1};
//    double training_out[] = {0.9};
    double error;
    double training_in[5], training_out[5];
    for (i = 0; i < 10; i++) {
        training_in[i] = normalize(11-i);
        training_out[i] = normalize((11-i) * (11-i));
    }

    initialize();


    gettimeofday(&t1, NULL);
    error = 1;
//    for (i = 0; i < 1000000; i++) {
    while (error > 1E-4) {
        error = 0;
        for (j = 0; j < 10; j++) {
            forward(&training_in[j]);
            error += (training_out[j] - outputs[0]) * (training_out[j] - outputs[0]);
//            printf("%2.0f^2 = %f\n", denormalize(training_in[j]), denormalize(outputs[0]));
            backpropagation(&training_in[j], &training_out[j]);
        }
        printf("Iteration:%lu  Error:%12.10f\n", it++, error);
    }
    gettimeofday(&t2, NULL);
    printf("Training time: %f\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-6);
    printf("Iterations: %lu\n", it);

    error = 0;
    for (i = 0; i < 10; i++) {
        forward(&training_in[i]);
        error += (training_out[i] - outputs[0]) * (training_out[i] - outputs[0]);
        printf("%2.0f^2 = %f\n", denormalize(training_in[i]), denormalize(outputs[0]));
    }
    printf("Total Error: %.3e\n", error);
    
    
    return 0;
}



