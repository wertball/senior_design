#ifndef header
#define header

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define c .5 //learning constant
#define I 1 //number of input layer nodes
#define H 3 //number of nodes per hidden layer
#define O 1 //number of output layer nodes
#define L 3 //number of layers

float weights[H][L-1]; //weights matrix
float h_o[H][L-2]; //hidden layer outputs

float feed_fwd(float x);
void initialize();
void train_network();
float activate(float x);
void multiply_accumulate();
void find_total_error();
void find_dw();
void update_w();
#endif


float feed_fwd(float x){
 	
  float input = activate(x);
  for(int i = 0; i < H; i++)
    h_o[i][0] = activate(input * weights[i][0]);
  	
  float sum = 0;
  for(int i = 0; i < H; i++)
    sum = sum + h[i][0] * weights[i][1];
  float output = activate(sum);
  return output;
}

void train_network(){
  	//feed_fwd
  	//find_total_error
  	//backpropagate
}
