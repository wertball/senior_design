#include "header_thomas.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

main(){
	float error, last, average, error_one, error_two;
	int i, j, k, kk;
	float input[] = {1, 2, 5, 9};
	float output[] = {1, 4, 25, 81};
	float input_bit[4];
	float output_bit[4];
	for(i = 0; i < 4; i++){
		input_bit[i] = input[i] / 256.0;
		output_bit[i] = output[i] / 256.0;
	}
	error_one = 0;
	initialize();
	printf("The initial weights are: \n");
	for(k = 0; k < L-1; k++){
		for(j = 0; j < H; j++)
			printf("%f  ",weights[j][k]);
	printf("\n");
	}
	//j = 2;
	for(j = 0; j < 4; j++){
		for(i = 0; i < 1000000; i++){
			error = train_network(input[j], output[j]);
			if(error < .000001)
				break;
		}
		printf("The total error is %f after %d runs\n", error, i);	
		printf("\n");
		last = (feed_fwd(input[j]));
		printf("With an input of %f, the output is %f, the desired output is %f, the difference is %f\n",input[j],last,output[j],last-output[j]);
		printf("The weights after run %d are: \n", (j+1));
		for(k = 0; k < L-1; k++){
			for(i = 0; i < H; i++){
				printf("%f  ",weights[i][k]);
				weights_one[i][k] = weights_one[i][k] + weights[i][k];
			}
		printf("\n");
		}
		error = 0;
	}
	for(k = 0; k < L-1; k++){
		for(i = 0; i < H; i++){
			weights[i][k] = weights_one[i][k] / 4;
		}
	}
	for(j = 0; j < 4; j++){
		last = (feed_fwd(input[j]));
		printf("With an input of %f, the output is %f, the desired output is %f, the difference is %f\n",input[j],last,output[j],last-output[j]);
	}
//	printf("The final weights are: \n");
//	for(k = 0; k < L-1; k++){
//		for(j = 0; j < H; j++)
//			printf("%f  ",weights[j][k]);
//	printf("\n");
//	}
//	printf("The total error is %f after %d runs\n", error, i);	
//	printf("\n");
	//j = 2;
//	for(j = 0; j < 4; j++){
//		last = (feed_fwd(input[j]));
//		printf("With an input of %f, the output is %f, the desired output is %f, the difference is %f\n",input[j],last,output[j],last-output[j]);
//	}
}

void initialize(){
	int i, j;
	int r;
	srand(time(NULL));
	for(i = 0; i < H; i++){
		for(j = 0; j < L-1; j++){
			r = rand() % 4;
			weights[i][j] = (float) r + 1.0;
			weights_new[i][j] = 0.0;
			if(j < L-2)
				h_o[i][j] = 0.0;
		}
	}
}

float feed_fwd(float x){
 	
	int i;
	float input, sum, output;
	input = x;
	//input -> hidden layer
	for(i = 0; i < H; i++)
		h_o[i][0] = activate(input * weights[i][0]);
	
	//hidden -> output layer
	sum = 0;
	for(i = 0; i < H; i++)
		sum = sum + h_o[i][0] * weights[i][1];
	output = sum;
	//large_output = output * 256.0;
	return output;
}

float train_network(float input, float output){
  	//feed_fwd
  	int i, j;
	float answer, dw, total_error, desired;
	answer = feed_fwd(input);
	//printf("answer is %f\n", answer);
	//desired = output / 256;
  	//find_total_error
	total_error = find_total_error((output), (answer));
	//printf("error is %f\n", total_error);
  	//backpropagate
	for(i = L-2; i > -1 ; i--){
		for(j = 0; j < H; j++){
			//calculate dw depeding on layer
			if(i == 0)
				dw = (answer - output) * weights[j][1] * h_o[j][0] * (1 - h_o[j][0]) * input;
			else
				dw = (answer - output) * h_o[j][i-1];
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

float deactivate(float y){
	return (-1 * log(1 / y - 1));
}

float find_total_error(float desired, float actual){
	return .5 * pow(desired - actual, 2);
}
