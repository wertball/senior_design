#ifndef NN_HEADER
#define NN_HEADER

#include <stdint.h>

typedef struct Nodes Node;
typedef struct Layers Layer;
typedef struct Neural_Nets Neural_Net;
typedef double calc_t;  //type used for algorithm computation

struct Nodes{
	uint8_t nw;	    //number of weights, up to 2^8 - 1
    calc_t o;       //output, node's output
    calc_t d;		//delta, dE/dni, affect of node's input on total error
    calc_t *cw;     //current weight, array of current weight values
    calc_t *uw;     //update weight, array of updated weight values
};

struct Layers{
    uint8_t nn;     //number of nodes, up to 2^8 - 1
    uint16_t nw;    //number of weights, up to 2^16 - 1
    Node **node;    //array of Layer nodes
    Layer *next;    //pointer to next layer
    Layer *prev;    //pointer to previous layer
};

struct Neural_Nets{
	uint8_t l;          //number of layers, up to 255
	uint32_t nn;        //number of nodes, up to 2^32 - 1
	uint32_t nw;        //number of weights, up to 2^32 - 1
    calc_t lc;          //learning constant
    calc_t cte;         //current total error on network
    Layer **layer;      //array of network layer addresses
};

typedef struct Neural_Net_Init_Parameters Neural_Net_Init_Params;
struct Neural_Net_Init_Parameters{
	uint8_t l;          //number of layers, [3:255]
    uint8_t *dim;       //dimensions of neural networ
    calc_t lc;          //learning constant
};

typedef struct Input_Set_Parameters Input_Set_Params;
struct Input_Set_Parameters{
    uint8_t set_size;       //number of input sets
    uint8_t input_set_size; //number of elements in a single input set
    calc_t min;             //minimum value of input set range
    calc_t max;             //maximum value of input set range
    calc_t step_size;       //step size between min and max values
};

void train_network(Neural_Net *nn, calc_t *input, calc_t output);
void feed_forward(Neural_Net *nn, calc_t *input);
void backpropagate(Neural_Net *nn, calc_t output);
void update_weights(Neural_Net *nn);
calc_t find_total_error(calc_t desired, calc_t actual);
calc_t percent_error(calc_t desired, calc_t actual);
calc_t activate(calc_t x);
calc_t* init_weights(int size);
Neural_Net* init_neural_net(Neural_Net_Init_Params *nnip);
calc_t* init_outputs(calc_t** input_set, uint8_t set_size);
calc_t** init_inputs(Input_Set_Params *isp);

#endif

