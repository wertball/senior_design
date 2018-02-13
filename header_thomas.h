#ifndef header_thomas
#define header_thomas

#define c .25 //learning constant
#define I 1 //number of input layer nodes
#define H 3 //number of nodes per hidden layer
#define O 1 //number of output layer nodes
#define L 3 //number of layers

float weights[H][L-1]; //weights matrix
float weights_new[H][L-1]; //new weights calculated by matrix
float weights_one[H][L-1];
float h_o[H][L-2]; //hidden layer outputs

float feed_fwd(float x);
float deactivate(float y);
void initialize();
float train_network(float input, float output);
float activate(float x);
float find_total_error(float desired, float actual);
#endif

