#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define INPUTS 1
#define H_LAYERS 1
#define H_HEIGHT 20
#define OUTPUTS 1
#define BIAS 1

#define DATA_MIN 1.0
#define DATA_MAX 82.0
#define DATA_RANGE (DATA_MAX - DATA_MIN)

//Weight declarations
float weights_in[INPUTS * H_HEIGHT];// = {.15,.20,.25,.30};
float weights_out[OUTPUTS * H_HEIGHT];// = {.65,.7,.75,.8};
#if (BIAS > 0)
float weights_bias_out[OUTPUTS];// = {0.85, 0.85};
float weights_bias_h[H_LAYERS][H_HEIGHT];// = {{.35,.35}, {.6,.6}};
#endif
#if (H_LAYERS > 1)
float weights_h[H_LAYERS - 1][H_HEIGHT * H_HEIGHT] = {.4,.45,.5,.55};
#endif

//I/O declarations
float h_out[H_LAYERS][H_HEIGHT];
float inputs[INPUTS]; //= {0.05, 0.10};
float outputs[OUTPUTS];

//Learning declarations
float ALPHA = 0.5;

//Randomization functions----------------------
void randomize_array(int length, float arr[length]) {
    for (int i = 0; i < length; i++) {
        arr[i] = ((rand() & 0b11) + 1) / ((rand() & 0xF) + 1);
        if ((rand() & 1) == 0) {
            arr[i] *= -1;
        }
    }
}

void randomize_2d_array(int rows, int columns, float arr[rows][columns]) {
    for (int i = 0; i < rows; i++) {
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
static inline float activation(float input) {
    return 1.0f / (1.0f + exp(-1 * input));
}

//Normalization functions--------------------
static inline float normalize(float input) {
    return (input - DATA_MIN) / (DATA_RANGE);
}

static inline float denormalize(float input) {
    return (DATA_RANGE) * input + DATA_MIN;
}
//--------------------------------------------

void forward(float in[INPUTS]) {
    //Input layer-----------------------
    for (int i = 0; i < H_HEIGHT; i++) {
        //Input weighted sum
        h_out[0][i] = 0;
        for (int j = 0; j < INPUTS; j++) {
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
    for (int i = 0; i < (H_LAYERS - 1); i++) { //For each layer
        for (int j = 0; j < H_HEIGHT; j++) { //For each node in a layer
            //Hidden weighted sum
            h_out[i+1][j] = 0;
            for (int k = 0; k < H_HEIGHT; k++) { //For each node in previous layer
                h_out[i+1][j] += weights_h[i][(j * H_HEIGHT) + k] * h_out[i][k];
            }
            
            //Bias weighted sum
            #if (BIAS > 0)
            h_out[i+1][j] += weights_bias_h[i+1][j] * BIAS;
            #endif
            
            //Activation
            h_out[i+1][j] = activation(h_out[i+1][j]);
            printf("h_out[%d][%d]: %f\n", i+1, j, h_out[i+1][j]);
        }
    }
    #endif
    //-----------------------------------------
    
    //Output Layer---------------------
    for (int i = 0; i < OUTPUTS; i++) {
        //Hidden layer weighted sum
        outputs[i] = 0;
        for (int j = 0; j < H_HEIGHT; j++) {
            outputs[i] += weights_out[(i * OUTPUTS) + j] * h_out[H_LAYERS - 1][j];
        }
        
        //Bias weighted sum
        #if (BIAS > 0)
        outputs[i] += weights_bias_out[i] * BIAS;
        #endif
        
        //Activation
        outputs[i] = normalize(outputs[i]);
        //printf("Output %d: %f\n", i, outputs[i]);
    }
    //-----------------------------------
}

void backpropagation(float in[INPUTS], float target_out[OUTPUTS]) {
    float delta_sum;
    float deltas_h[H_HEIGHT];
    float deltas_o[OUTPUTS];
    float weights_out_new[OUTPUTS * H_HEIGHT];
    #if (H_LAYERS > 1)
    //TODO: Make this more memory efficient
    float weights_h_new[H_LAYERS - 1][H_HEIGHT * H_HEIGHT];
    #endif

    
    //Output layer------------------------------
    for (int i = 0; i < OUTPUTS; i++) {
        deltas_o[i] = (outputs[i] - target_out[i]);// * (outputs[i] * (1 - outputs[i]));
        for (int j = 0; j < H_HEIGHT; j++) {
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
    //TODO: Output Layer
    //Layer connected to outputs
    for (int i = 0; i < H_HEIGHT; i++) { //For each node in hidden layer
        delta_sum = 0;
        for (int j = 0; j < OUTPUTS; j++) {  //For each output
            delta_sum += weights_out[i + (j * OUTPUTS)] * deltas_o[j];
        }
        deltas_h[i] = (h_out[H_LAYERS-1][i] * (1 - h_out[H_LAYERS-1][i])) * delta_sum;
        for (int j = 0; j < H_HEIGHT; j++) {
            weights_h_new[H_LAYERS-1][(i * H_HEIGHT) + j] = weights_h[H_LAYERS-1][(i * H_HEIGHT) + j] -
                    (ALPHA * deltas_h[i] * h_out[H_LAYERS-2][j]);
            printf("New weight_h[%d][%d]: %f\n", H_LAYERS-2, (i * H_HEIGHT) + j, weights_h_new[H_LAYERS-2][(i * H_HEIGHT) + j]);
        }
        #if (BIAS > 0)
        weights_bias_h[H_LAYERS-2][i] -= ALPHA * deltas_h[i] * BIAS;
        #endif
    }
    //TODO: Hidden Layers
    //Each hidden layer not connected to input/hidden output layer
    for (int i = H_LAYERS-2; i > 1; i--) {
        for (int j = 0; j < H_HEIGHT; j++) {
            delta_sum = 0;
            for (int k = 0; k < OUTPUTS; k++) {
                delta_sum += weights_out[i + (k * OUTPUTS)] * deltas_o[k];
            }
            deltas_h[i] = (h_out[0][i] * (1 - h_out[0][i])) * delta_sum;
            for (int k = 0; k < INPUTS; k++) {
                weights_in[(i * INPUTS) + k] -= (ALPHA * deltas_h[i] * in[k]);
                printf("New weight_h[%d][%d]: %f\n", , , );
            }
            #if (BIAS > 0)
            weights_bias_h[0][i] -= ALPHA * deltas_h[i] * BIAS;
            #endif
        }
    }
    //TODO: Input Layer
    //Layer connected to inputs
    for (int i = 0; i < H_HEIGHT; i++) {
        delta_sum = 0;
        for (int j = 0; j < OUTPUTS; j++) {
            delta_sum += weights_out[i + (j * OUTPUTS)] * deltas_o[j];
        }
        deltas_h[i] = (h_out[0][i] * (1 - h_out[0][i])) * delta_sum;
        for (int j = 0; j < INPUTS; j++) {
            weights_in[(i * INPUTS) + j] -= (ALPHA * deltas_h[i] * in[j]);
            printf("New weight_in[%d]: %f\n", (i * INPUTS) + j, weights_in[(i * INPUTS) + j]);
        }
        #if (BIAS > 0)
        weights_bias_h[0][i] -= ALPHA * deltas_h[i] * BIAS;
        #endif
    }
    #else
    for (int i = 0; i < H_HEIGHT; i++) {
        delta_sum = 0;
        for (int j = 0; j < OUTPUTS; j++) {
            delta_sum += weights_out[i + (j * OUTPUTS)] * deltas_o[j];
        }
        deltas_h[i] = (h_out[0][i] * (1 - h_out[0][i])) * delta_sum;
        for (int j = 0; j < INPUTS; j++) {
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
    for (int i = 0; i < OUTPUTS * H_HEIGHT; i++) {
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



int main() {
    float training_in[41], training_out[41];
	float testing_in[40], testing_out[40];
    for (int i = 0; i < 41; i++) {
        training_in[i] = normalize(i * 2.0 + 1.0);
        training_out[i] = normalize(sqrtf(i*2.0+1.0));
    }
	for (int i = 1; i < 40; i++){
		testing_in[i] = normalize(i*2.0);
		testing_out[i] = normalize(sqrtf(i*2.0));
	}

    initialize();

    for (int i = 0; i < 1000000; i++) {
        for (int j = 0; j < 41; j++) {
            forward(&training_in[j]);
            backpropagation(&training_in[j], &training_out[j]);
        }
    }

    float error = 0;
    for (int i = 0; i < 41; i++) {
        forward(&training_in[i]);
        error += 0.5 * powf((denormalize(training_out[i]) - denormalize(outputs[0])),2);
        printf("sqrt(%f) = %f\n", denormalize(training_in[i]), denormalize(outputs[0]));
    }
    printf("Train Error: %f\n", error/41);
    
	error = 0;
	for (int i = 0; i < 40; i++) {
        forward(&testing_in[i]);
        error += 0.5 * powf((denormalize(testing_out[i]) - denormalize(outputs[0])),2);
        printf("sqrt(%f) = %f\n", denormalize(testing_in[i]), denormalize(outputs[0]));
    }
    printf("Test Error: %f\n", error/41);
    
    
    return 0;
}



