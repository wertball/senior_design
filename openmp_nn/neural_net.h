#ifndef NN_HEADER
#define NN_HEADER

#include <stdint.h>

typedef struct Nodes Node;
typedef struct Layers Layer;
typedef struct Neural_Nets Neural_Net;
typedef double calc_t;  //type used for algorithm computation
typedef struct ThreadData Thread_Data; //unique resources need per thread

struct ThreadData{
    calc_t o;       //output, node's output
    calc_t d;		//delta, dE/dni, affect of node's input on total error
    calc_t *uw;     //update weight, array of updated weight values
};

struct Nodes{
	uint16_t nw;	            //number of weights, up to 2^8 - 1
	calc_t *cw;             //current weight, array of current weight values
    Thread_Data **thread;   //array of ThreadData pointers, 1 for every thread
};

struct Layers{
	uint16_t nn;     //number of nodes, up to 2^16 - 1
	uint16_t nw;    //number of weights, up to 2^16 - 1
	Node **node;   //Layer->Node[node_index][thread_id]
	Layer *next;    //pointer to next layer
	Layer *prev;    //pointer to previous layer
};

struct Neural_Nets{
	uint8_t l;          //number of layers, up to 255
    uint8_t nt;         //number of threads, up to 255
	uint32_t nn;        //number of nodes, up to 2^32 - 1
	uint32_t nw;        //number of weights, up to 2^32 - 1
	calc_t lc;          //learning constant
	calc_t cte;         //current total error on network
	Layer **layer;      //array of network layer addresses
};

typedef struct Neural_Net_Init_Parameters Neural_Net_Init_Params;
struct Neural_Net_Init_Parameters{
	uint8_t l;          //number of layers, [3:255]
	uint8_t *dim;       //dimensions of neural network
    uint8_t nthreads;   //number of threads expected to execute nn with
	calc_t lc;          //learning constant
};

calc_t train_network(Neural_Net *nn, calc_t *input, calc_t output);
void feed_forward(Neural_Net *nn, calc_t *input, int tid, calc_t *data_range);
void backpropagate(Neural_Net *nn, calc_t output, int tid);
void sync_update_weights(Neural_Net *nn, calc_t avg_divisor);
calc_t find_total_error(calc_t desired, calc_t actual);
calc_t percent_error(calc_t desired, calc_t actual);
calc_t activate(calc_t x);
calc_t* init_weights(int size, calc_t bound);
Neural_Net* init_neural_net(Neural_Net_Init_Params *nnip);
calc_t normalize(calc_t input, calc_t *data_range);
calc_t denormalize(calc_t input, calc_t *data_range);

//help functions
void printWeights(Neural_Net *nn);
void printNetwork(Neural_Net *nn);

#endif

