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

calc_t train_network(Neural_Net *nn, calc_t *input, calc_t output);
void feed_forward(Neural_Net *nn, calc_t *input);
void backpropagate(Neural_Net *nn, calc_t output);
void update_weights(Neural_Net *nn);
calc_t find_total_error(calc_t desired, calc_t actual);
calc_t percent_error(calc_t desired, calc_t actual);
calc_t activate(calc_t x);
calc_t* init_weights(int size, calc_t bound);
Neural_Net* init_neural_net(Neural_Net_Init_Params *nnip);
calc_t normalize(calc_t input, calc_t min, calc_t max);
calc_t denormalize(calc_t input, calc_t min, calc_t max);

//help functions
void printWeights(Neural_Net *nn);

#endif

