#ifndef header_thomas
#define header_thomas

#define c .25 //learning constant
#define I 1 //number of input layer nodes
#define H 10 //number of nodes per hidden layer
#define O 1 //number of output layer nodes
#define L 3 //number of layers
#define training_set_size 4 //size of training set
#define testing_set_size 4 //size of testing set

float weights[H][L-1]; //weights matrix
float weights_new[H][L-1]; //new weights calculated by matrix
float weights_one[H][L-1];
float h_o[H][L-2]; //hidden layer outputs

#ifdef BIAS
float bias[H];
#endif

float training_set_input[training_set_size] = {.1,.3,.7,.9};
float training_set_target[training_set_size] = {.01,.09,.49,.81};

float testing_set_input[testing_set_size] = {.2,.4,.6,.8};
float testing_set_target[testing_set_size] = {.04,.16,.36,.64};

float feed_fwd(float x);
float deactivate(float y);
void initialize();
float train_network(float input, float output);
float activate(float x);
float find_total_error(float desired, float actual);
#endif

