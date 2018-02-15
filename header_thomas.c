#include "header_thomas.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

main(){

	float error, last, average, error_one, error_two;
	int i, j, k, kk;
	//float input[] = {1, 2, 5, 6};
	//float output[] = {1, 4, 25, 36};
	//float input_bit[4];
	//float output_bit[4];
	//for(i = 0; i < 4; i++){
	//	input_bit[i] = input[i] / 256.0;
	//	output_bit[i] = output[i] / 256.0;
	//}
	error_one = 0;

	initialize();

#ifdef DEBUG_MODE
	printf("The initial weights are: \n");
	for(k = 0; k < L-1; k++){
		for(j = 0; j < H; j++)
			printf("%f  ",weights[j][k]);
	printf("\n");
	}

#ifdef BIAS
	printf("The initial biases are: \n");
	for(k = 0; k < H; k++)
		printf("%f ",bias[k]);
	printf("\n");
#endif
#endif


	for(i = 0; i < 1000000; i++){
		for(j = 0; j < training_set_size; j++){
			error = train_network(training_set_input[j], training_set_target[j]);
			error_one += error;
		}
		if((error_one / training_set_size) < .000001)
			break;
		error_one = 0;
	}

	printf("The total error is %f after %d runs\n", error, i);

	for(j = 0; j < training_set_size; j++){
		last = (feed_fwd(training_set_input[j]));
		printf("With an input of %f, the output is %f, the desired output is %f, the difference is %f\n",training_set_input[j],last,training_set_target[j],last-training_set_target[j]);
#ifdef DEBUG_MODE
		printf("The outputs are: \n");
		for(k = 0; k < H; k++){
			printf("%f ", h_o[k][0]);
		}
		printf("\n");
#endif
	}

	printf("\n");
	error_one = 0;
	for(j = 0; j < testing_set_size; j++){
		last = (feed_fwd(testing_set_input[j]));
		error = find_total_error(testing_set_target[j], last);
		printf("With an input of %f, the output is %f, the desired output is %f, the difference is %f\n",testing_set_input[j],last,testing_set_target[j],last - testing_set_target[j]);
		error_one += error;
	}
	printf("The total error of the testing set is %f\n", error_one/testing_set_size);

#ifdef DEBUG_MODE
	printf("The final weights are: \n");
	for(k = 0; k < L-1; k++){
		for(j = 0; j < H; j++)
			printf("%f  ",weights[j][k]);
	printf("\n");
	}

#ifdef BIAS
	printf("The final biases are: \n");
	for(k = 0; k < H; k++)
		printf("%f ",bias[k]);
	printf("\n");
#endif
#endif

}

void initialize(){
	int i, j;
	int r;

	srand(time(NULL));

	for(i = 0; i < H; i++){
		for(j = 0; j < L-1; j++){
			r = rand() % 4;
			weights[i][j] = (float) r + 1;
			weights_new[i][j] = 0.0;
			if(j < L-2){
				h_o[i][j] = 0.0;
#ifdef BIAS
				bias[i] = (float) r + 2;
#endif
			}
		}
	}

}

float feed_fwd(float x){
 	
	int i;
	float input, sum, output;
	input = x;

	//input -> hidden layer
	for(i = 0; i < H; i++)
		h_o[i][0] = activate(input * weights[i][0] 
#ifdef BIAS
								+ bias[i]
#endif
							);
	
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
	float answer, dw, total_error, desired, bw, db;
	answer = feed_fwd(input);

  	//find_total_error
	total_error = find_total_error(output, answer);

  	//backpropagate
	for(i = L-2; i > -1 ; i--){
		for(j = 0; j < H; j++){
			//calculate dw depeding on layer
			if(i == 0)
				dw = (answer - output) * weights[j][1] * h_o[j][0] * (1 - h_o[j][0]) * input;
			else
				dw = (answer - output) * h_o[j][i-1];
			//store new weights
			//printf("answer is %f, output is %f, error is %f, weight is %f, h_o is %f, input is %f, dw is %f\n", answer,output,total_error,weights[j][i],h_o[j][0],input,dw);
			weights_new[j][i] = weights[j][i] - dw * c;
		}
		//printf(".\n");
	}
#ifdef BIAS
	for(j = 0; j < H; j++){
		bw = bias[j];
		db = (answer - output) * weights[j][1] * h_o[j][0] * (1 - h_o[j][0]);
		bw = bw - db * c;
		bias[j] = bw;
	}
#endif

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
