#include "neural_net.h"
#include "debug.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define num_layers 3
#define lc 0.1
#define training_iterations 1e7
#define target_error 1e-20

#define training_set_size 25
#define set_input_size 2
#define input_set_lower_bound 0.1
#define input_set_upper_bound 0.5
#define input_set_step_size 0.1

 int main(void){
    //network initialization parameters
    uint8_t dim[num_layers] = {
            set_input_size,  //layer 1, input layer
            //5,              //layer 2, hidden layer
            //5,              //layer 3, hidden layer
            5,              //layer 4, hidden layer
            1               //layer 5, output layer
    };	//number of nodes in each layer
    Neural_Net_Init_Params *nnip = &(Neural_Net_Init_Params){num_layers, dim, lc};

    //create and initialize neural net
    Neural_Net *nn = init_neural_net(nnip);
    printWeights(nn);

    //create input and output arrays, print during debug
    Input_Set_Params *isp = &(Input_Set_Params) {
             training_set_size,         //number of input sets
             set_input_size,            //number of elements in a single input set
             input_set_lower_bound,     //minimum value of input set range
             input_set_upper_bound,     //maximum value of input set range
             input_set_step_size        //step size between min and max values
    };
    //calc_t** input_sets = init_inputs(isp);
    calc_t input_sets[training_set_size][set_input_size] = {
            {0.5, 0.5},
            {0.5, 0.4},
            {0.5, 0.3},
            {0.5, 0.2},
            {0.5, 0.1},
            {0.4, 0.5},
            {0.4, 0.4},
            {0.4, 0.3},
            {0.4, 0.2},
            {0.4, 0.1},
            {0.3, 0.5},
            {0.3, 0.4},
            {0.3, 0.3},
            {0.3, 0.2},
            {0.3, 0.1},
            {0.2, 0.5},
            {0.2, 0.4},
            {0.2, 0.3},
            {0.2, 0.2},
            {0.2, 0.1},
            {0.1, 0.5},
            {0.1, 0.4},
            {0.1, 0.3},
            {0.1, 0.2},
            {0.1, 0.1}
    };
    calc_t output[training_set_size] = {
            1.0,
            0.94868329805,
            0.894427191,
            0.83666002653,
            0.77459666924,

            0.94868329805,
            0.894427191,
            0.83666002653,
            0.77459666924,
            0.70710678118,

            0.894427191,
            0.83666002653,
            0.77459666924,
            0.70710678118,
            0.632455532,

            0.83666002653,
            0.77459666924,
            0.70710678118,
            0.632455532,
            0.547722558,

            0.77459666924,
            0.70710678118,
            0.632455532,
            0.547722558,
            0.447213595
    };
    //init_inputs(~)
    //init_outputs(~)
    //printInputs(isp, input_sets);
    //printOutputs(isp, output);

    //train it
    int i;
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for(i = 0; i < training_iterations && nn->cte > target_error; i++) {
        for(int j = 0; j < training_set_size; j++) {
            train_network(nn, input_sets[j], output[j]);
        }
    }
    gettimeofday(&t2, NULL);
    printf("execution time(s): %.4f\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-6);
    printf("total error: %e\n", nn->cte);
    printf("total runs = %d\n", i);

    //test network
    calc_t input_set[isp->input_set_size];
    for(calc_t x = 0.05; x <= 0.55; x += 0.05){
        for(calc_t y = 0.05; y <= 0.55; y += 0.05){
            input_set[0] = x;
            input_set[1] = y;
            feed_forward(nn, input_set);
            calc_t output = nn->layer[num_layers - 1]->node[0]->o;
            DEBUG_PRINT(("Input: (%.2f, %.2f) => %.2e ", x, y, output));
            DEBUG_PRINT(("Error = %.2e%\n", percent_error(sqrt(x + y), output)));
        }
    }
    free(nn); //free(input_sets); free(output);
    return 0;
}

