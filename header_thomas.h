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
float weights_new[H][L-1]; //new weights calculated by matrix
float h_o[H][L-2]; //hidden layer outputs

float feed_fwd(float x);
void initialize();
void train_network(float input, float output);
float activate(float x);
float find_total_error(float desired, float actual);
#endif

void initialize(){
	for(int i = 0; i < H; i++){
		for(int j = 0; j < L-1; j++){
			weights[i][j] = 0.0;
			weights_new[i][j] = 0.0;
			if(j < L-2)
				h_o[i][j] = 0.0;
		}
	}
}

float feed_fwd(float x){
 	
	float input = activate(x);
	//input -> hidden layer
	for(int i = 0; i < H; i++)
		h_o[i][0] = activate(input * weights[i][0]);
	
	//hidden -> output layer
	float sum = 0;
	for(int i = 0; i < H; i++)
		sum = sum + h_o[i][0] * weights[i][1];
	float output = activate(sum);
	return output;
}

void train_network(float input, float output){
  	//feed_fwd
	float answer;
	answer = feed_fwd(input);
  	//find_total_error
	float total_error = find_total_error(output, answer);
  	//backpropagate
	for(int i = L-2; i > -1 ; i++){
		for(int j = 0; j < H; j++){
			//calculate dw depeding on layer
			if(i == 0)
				float dw = (answer - output) * answer * (1 - answer) * weights[j][1] * h_o[j][0] * (1 - h_o[j][0]) * activate(input);
			else
				float dw = (answer - output) * answer * (1 - answer) * h_o[j][i-1];
			//store new weights
			weights_new[j][i] = weights[j][i] - dw * c;
		}
	}
	//implement new weights
	weights = weights_new;
}

float activate(float x){
	return 1.0 / (1.0 + exp(-1 * x));
}

float find_total_error(float desired, float actual){
	return .5 * pow(desired - actual, 2);
}