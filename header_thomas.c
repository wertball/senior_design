#include "header_thomas.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

main(){
	float error, last;
	int i, j;
	float input[] = {1, .2, 5};
	float output[] = {1, .4, 25};
	initialize();
	for(i = 0; i < 100000; i++){
		//for(j = 0; j < 3; j++){
			error = train_network(input[1], output[1]);
			if(error < .000001)
				break;
		//}
	}
	printf("The total error is %f after %d runs\n", error, i);
	//for(i = 0; i < 3; i++){
		last = feed_fwd(input[1]);
		printf("With an input of %f, the output is %f\n",input[1],last);
	//}
}

void initialize(){
	int i, j;
	for(i = 0; i < H; i++){
		for(j = 0; j < L-1; j++){
			weights[i][j] = 1.0;
			weights_new[i][j] = 0.0;
			if(j < L-2)
				h_o[i][j] = 0.0;
		}
	}
}

float feed_fwd(float x){
 	
	int i;
	float input, sum, output;
	input = activate(x);
	//input -> hidden layer
	for(i = 0; i < H; i++)
		h_o[i][0] = activate(input * weights[i][0]);
	
	//hidden -> output layer
	sum = 0;
	for(i = 0; i < H; i++)
		sum = sum + h_o[i][0] * weights[i][1];
	output = activate(sum);
	return output;
}

float train_network(float input, float output){
  	//feed_fwd
  	int i, j;
	float answer, dw, total_error;
	answer = feed_fwd(input);
  	//find_total_error
	total_error = find_total_error(output, answer);
  	//backpropagate
	for(i = L-2; i > -1 ; i--){
		for(j = 0; j < H; j++){
			//calculate dw depeding on layer
			if(i == 0)
				dw = (answer - output) * answer * (1 - answer) * weights[j][1] * h_o[j][0] * (1 - h_o[j][0]) * activate(input);
			else
				dw = (answer - output) * answer * (1 - answer) * h_o[j][i-1];
			//store new weights
			weights_new[j][i] = weights[j][i] - dw * c;
		}
	}
	//implement new weights
	for(i = 0; i < L-1; i++){
		for(j = 0; j < H; j++)
			weights[j][i] = weights_new[j][i];
	}
	return total_error;
}

float activate(float x){
	return 1.0 / (1.0 + exp(-1 * x));
}

float find_total_error(float desired, float actual){
	return .5 * pow(desired - actual, 2);
}
