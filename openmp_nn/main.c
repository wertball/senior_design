#include "neural_net.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <errno.h>
#include <omp.h>
#include <string.h>
#include <float.h>
//works on gcc 6.4.0

//Flags---------------------------------------------------------
//#define DEBUGGING //comment this flag to disable debug messages
#define FILE_IO   //comment this flag to disable file IO

#ifdef DEBUGGING
# define DEBUG(x) x
#else
# define DEBUG(x) do {} while (0)
#endif

#ifdef FILE_IO
# define FIO(x) x
#else
# define FIO(x) do {} while (0)
#endif
//--------------------------------------------------------------

//Network Parameters--------------
#define num_layers 5
#define learning_rate 0.2
#define training_iterations 1000000
#define target_error 9e1
#define data_min 0.520171
#define data_max 3.43917

#define training_set_size 55
#define set_input_size 5
//Network Parameters--------------

//File Locations------------------------------------------------------------------------
#define file_base_directory "F:\\school\\ELEC4000\\senior_design\\gitRepo\\thomas_nn\\"
#define test_output_file ("test_output.txt")
#define error_results_file ("Error_Results.txt")
#define test_results_file ("Test_Results.txt")
//--------------------------------------------------------------------------------------

int read_data(char *file_name, calc_t ***training_in, calc_t **training_out, calc_t ***data_range);
void normalizeIOSets(int samples, calc_t **t_in, calc_t *t_out, calc_t **d_range);

int main(int argc, char **argv){
    printf("---------------------------------------------------------------------------------------------\n");
    //parse input - main {num_threads | training_results_file | testing_results_file}
    uint8_t num_threads = 1;
    char *testing_results_file = (file_base_directory"Testing_Results.txt"),
         *training_results_file = (file_base_directory"Training_Results.txt");
    if(argc >= 2)
        num_threads = strtol(argv[1], NULL, 0);
    if(argc >= 4){
        training_results_file = argv[2];
        testing_results_file = argv[3];
    }
    printf("Attempting to run with %d threads . . .\n", num_threads);
    printf("Training Results File: %s\n", training_results_file);
    printf("Testing Results File: %s\n", testing_results_file);

    //timing variables
    struct timeval t1, t2;

    //openmp setup
    omp_set_num_threads(num_threads);

    //network initialization parameters
	uint8_t dim[num_layers] = {
			set_input_size,  //layer 1, input layer
			20,              //layer 2, hidden layer
            20,              //layer 3, hidden layer
            20,              //layer 4, hidden layer
			//5,				//layer 5, hidden layer
			1               //layer 6, output layer
	};	//number of nodes in each layer
	Neural_Net_Init_Params *nnip = &(Neural_Net_Init_Params){num_layers, dim, num_threads, learning_rate};

	//create and initialize neural net
	Neural_Net *nn = init_neural_net(nnip);
	DEBUG(printNetwork(nn););

    //input and output training data setup
    calc_t **input_sets;
    calc_t *output;
    calc_t **data_range;
    int training_set_size = read_data(training_data_file, &input_sets, &output, &data_range);
    FIO(printf("Number of training data samples read: %d\n", training_set_size););
    normalizeIOSets(training_set_size, input_sets, output, data_range);

	//train the network
	int i, j, tid;
	calc_t error = 1.0;
    FILE *fp;
    fp = fopen(training_results_file,"w");
    if(fp == NULL)
        printf("Failed to write %s\nerrno: %d\n", training_results_file, errno);

    gettimeofday(&t1, NULL);
	//train using percent error calculation--------------------------
	for(i = 0; i < training_iterations /*&& error > target_error*/; i++){
		error = 0.0;
        //assume update weights to be zero
	    #pragma omp parallel for private(tid, error_1) reduction(+:error)
        for(j = 0; j < training_set_size; j++){
            tid = omp_get_thread_num();

            //training
            feed_forward(nn,input_sets[j], tid, data_range[set_input_size]);
            backpropagate(nn, output[j], tid);

            //determine expected and actual values
            calc_t expected = denormalize(output[j], data_range[set_input_size]);
            calc_t actual = denormalize(nn->layer[nn->l-1]->node[0]->thread[tid]->o, data_range[set_input_size]);

            //determining percent error
            error += percent_error(expected, actual);

        }
        sync_update_weights(nn, training_set_size);
		error /= training_set_size;
		fprintf(fp, "%d\t%.2e\n", i, error);
	}
	//---------------------------------------------------------------

    //record and print training results------------------------------------------------------------
	gettimeofday(&t2, NULL);
	fclose(fp);
	printf("\ntraining time(s): %.4f\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-6);
	printf("total error: %e\n", error);
	printf("total runs = %d\n\n", i);
    //---------------------------------------------------------------------------------------------

	//test network
    //input and output test data setup
    calc_t **test_input_sets;
    calc_t *test_output;
    calc_t **test_data_range;
    int testing_set_size = read_data(testing_data_file, &test_input_sets, &test_output, &test_data_range);
    FIO(printf("Number of training data samples read: %d\n", testing_set_size););
    normalizeIOSets(testing_set_size, test_input_sets, test_output, data_range);

    calc_t test_error = 0;
    calc_t largest = 0.0;
    calc_t smallest = 100.0;
    fp = fopen(testing_results_file,"w");
    if(fp == NULL)
        printf("Failed to write %s\nerrno: %d", testing_results_file, errno);

    tid = omp_get_thread_num();
	for(i = 0; i < testing_set_size; i++){
        //feed test data
		feed_forward(nn, test_input_sets[i], tid, data_range[set_input_size]);

        //determine expected and actual values
        calc_t expected = denormalize(test_output[i], data_range[set_input_size]);
        calc_t actual = denormalize(nn->layer[nn->l-1]->node[0]->thread[tid]->o, data_range[set_input_size]);

        //print results to file
		fprintf(fp,"%.2f, %.2f, ", actual, expected);
		error = percent_error(expected, actual);
		fprintf(fp,"Error = %.2f\n", error);

        //characterize data
		test_error += error;
        smallest = (error < smallest) ? error : smallest;
        largest = (error > largest) ? error : largest;
	}
	fclose(fp);
    printf("Test Error: \n");
    printf("smallest error: %.4f\n", smallest);
    printf("largest error: %.4f\n", largest);
    printf("range of errors: %.4f\n", largest - smallest);
	printf("average error: %.2f\n", test_error/testing_set_size);

	//dealloc neural network
	//dealloc(nn);
	return 0;
}

int read_data(char *file_name, calc_t ***training_in, calc_t **training_out, calc_t ***data_range) {
    int i, j, samples, current_line;
    calc_t **t_in, *t_out, **d_range;
    char temp_str[100];
    char *pointer;
    FILE *fp;

    //Open file
    fp = fopen(file_name, "r");
    if (!fp) {
        printf("Error: Failed to open %s\n", file_name);
        fflush(stdout);
        return -1;
    }

    //count number of samples
    samples = 0;
    while (fgets(temp_str, 100, fp) != NULL) {
        if (strlen(temp_str) > 2) {
            samples++;
        }
    }
    if (samples == 0) {
        printf("Error: No data in %s\n", file_name);
        fflush(stdout);
        return -1;
    }
    rewind(fp);

    //Allocate arrays
    t_in = (calc_t **) malloc(samples * sizeof(calc_t *));
    d_range = (calc_t **) malloc((set_input_size + 1) * sizeof(calc_t *));
    for (i = 0; i < samples; i++) {
        t_in[i] = (calc_t *) malloc(set_input_size * sizeof(calc_t));
    }
    for (i = 0; i < (set_input_size + 1); i++) {
        d_range[i] = (calc_t *) malloc(2 * sizeof(calc_t));
        d_range[i][0] = DBL_MAX;
        d_range[i][1] = DBL_MIN;
    }
    t_out = (calc_t *) malloc(samples * sizeof(calc_t));
    *training_in = t_in;
    *training_out = t_out;
    *data_range = d_range;

    //Get data from file
    i = 0;
    current_line = 1;
    while (fgets(temp_str, 100, fp) != NULL) {
        if (strlen(temp_str) > 2) {
            pointer = strtok(temp_str, "\t");
            for (j = 0; j < set_input_size; j++) {
                if (pointer == NULL) {
                    printf("Error: Unexpected end of line: %d\n", current_line);
                    fflush(stdout);
                    return -1;
                }
                sscanf(pointer, "%lf", &t_in[i][j]);
                if (t_in[i][j] < d_range[j][0]) {
                    d_range[j][0] = t_in[i][j];
                }
                if (t_in[i][j] > d_range[j][1]) {
                    d_range[j][1] = t_in[i][j];
                }
                pointer = strtok(NULL, "\t");
            }
            if (pointer == NULL) {
                printf("Error: Unexpected end of line: %d\n", current_line);
                fflush(stdout);
                return -1;
            }
            sscanf(pointer, "%lf", &t_out[i]);
            if (t_out[i] < d_range[set_input_size][0]) {
                d_range[set_input_size][0] = t_out[i];
            }
            if (t_out[i] > d_range[set_input_size][1]) {
                d_range[set_input_size][1] = t_out[i];
            }
            //t_out[i] = normalize(t_out[i]);
            i++;
        }
        current_line++;
    }
    //close file
    fclose(fp);

    return samples;
}

void normalizeIOSets(int samples, calc_t **t_in, calc_t *t_out, calc_t **d_range){
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < set_input_size; j++) {
            if (d_range[j][0] == d_range[j][1]) {
                t_in[i][j] = 1;
            }
            else {
                t_in[i][j] = normalize(t_in[i][j], d_range[j]);
            }
        }
        t_out[i] = normalize(t_out[i], d_range[set_input_size]);
    }
}

