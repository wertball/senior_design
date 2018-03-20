#include "neural_net.h"
//#include "debug.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <errno.h>

//Flags---------------------------------------------------------
#define DEBUGGING //comment this flag to disable debug messages
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
#define training_iterations 1e7
#define target_error 1e-6
#define data_min 0.549361
#define data_max 3.43917

#define training_set_size 55
#define set_input_size 5
#define testing_set_size 360
#define input_set_lower_bound 0.1
#define input_set_upper_bound 0.5
#define input_set_step_size 0.1
//Network Parameters--------------

//File Locations------------------------------------------------------------------------
#define file_base_directory "F:\\school\\ELEC4000\\senior_design\\gitRepo\\thomas_nn\\"
#define test_output_file (file_base_directory"test_output.txt")
#define error_results_file (file_base_directory"Error_Results.txt")
#define test_results_file (file_base_directory"Test_Results.txt")
//--------------------------------------------------------------------------------------

 int main(void){
    FILE *fp;

    //network initialization parameters
    uint8_t dim[num_layers] = {
            set_input_size,  //layer 1, input layer
            20,              //layer 2, hidden layer
            20,              //layer 3, hidden layer
            20,              //layer 4, hidden layer
            //5,				//layer 5, hidden layer
            1               //layer 6, output layer
    };	//number of nodes in each layer
    Neural_Net_Init_Params *nnip = &(Neural_Net_Init_Params){num_layers, dim, learning_rate};

    //create and initialize neural net
    Neural_Net *nn = init_neural_net(nnip);
    DEBUG(printWeights(nn);)

    //create input and output arrays, print during debug
/*    Input_Set_Params *isp = &(Input_Set_Params) {
             training_set_size,         //number of input sets
             set_input_size,            //number of elements in a single input set
             input_set_lower_bound,     //minimum value of input set range
             input_set_upper_bound,     //maximum value of input set range
             input_set_step_size        //step size between min and max values
    };*/

    //Create input data--------------------------------------------------------------------
    calc_t denormalized_input_sets[training_set_size][set_input_size];
    for(int i = 0; i < training_set_size; i++){
        denormalized_input_sets[i][0] = (calc_t) 10.0;
        denormalized_input_sets[i][1] = (calc_t) 20.0;
        denormalized_input_sets[i][2] = (calc_t) 300.0;
        denormalized_input_sets[i][3] = (calc_t) ((i / 11) * 10.0 + 80.0);
        denormalized_input_sets[i][4] = (calc_t) (1.79 + (i % 11)/100.0);
    }
    //-------------------------------------------------------------------------------------

    //Normalize input data-----------------------------------------------------------------
    calc_t input_sets[training_set_size][set_input_size];
    for(int i = 0; i < training_set_size; i++){
        input_sets[i][0] = denormalized_input_sets[i][0] / 10.0;
        input_sets[i][1] = denormalized_input_sets[i][1] / 20.0;
        input_sets[i][2] = denormalized_input_sets[i][2] / 300.0;
        input_sets[i][3] = (denormalized_input_sets[i][3] - 80.0) / (120.0 - 80.0);
        input_sets[i][4] = (denormalized_input_sets[i][4] - 1.79) / (1.89 - 1.79);
        //for(int j = 0; j < set_input_size; j++i)
        //	input_sets[i][j] = normalize(denormalized_input_sets[i][j]);
    }
    //-------------------------------------------------------------------------------------

    //Create output data--------------------------------------------------------------------
    calc_t denormalized_output[training_set_size] = {
        3.43313,
        2.71998,
        2.23442,
        1.87542,
        1.57747,
        1.32098,
        1.10156,
        0.921271,
        0.782483,
        0.681796,
        0.610265,
        3.43917,
        2.78111,
        2.25032,
        1.86824,
        1.55482,
        1.30133,
        1.07933,
        0.897866,
        0.75942,
        0.658494,
        0.587405,
        2.98477,
        2.49611,
        2.12776,
        1.78697,
        1.49209,
        1.2415,
        1.02859,
        0.858088,
        0.727433,
        0.633834,
        0.566431,
        3.30783,
        2.64033,
        2.13554,
        1.78793,
        1.5096,
        1.25894,
        1.04158,
        0.865621,
        0.731832,
        0.635492,
        0.567038,
        2.901,
        2.35628,
        1.98702,
        1.70146,
        1.43822,
        1.19831
    };
    //--------------------------------------------------------------------------------------

    calc_t denormalized_test_input_sets[testing_set_size][set_input_size];
    for(int i = 0; i < testing_set_size; i++){
        denormalized_test_input_sets[i][0] = (calc_t) 10.0;
        denormalized_test_input_sets[i][1] = (calc_t) 20.0;
        denormalized_test_input_sets[i][2] = (calc_t) 300.0;
        denormalized_test_input_sets[i][3] = (calc_t) (81.0 + (i / 90 * 10) + ((i % 90) / 10));
        denormalized_test_input_sets[i][4] = (calc_t) (1.795 + (i % 10)/100.0);
    }

    //Normalize output data-----------------------------------------------------------------
    calc_t test_input_sets[testing_set_size][set_input_size];
    for(int i = 0; i < testing_set_size; i++){
        test_input_sets[i][0] = denormalized_test_input_sets[i][0] / 10.0;
        test_input_sets[i][1] = denormalized_test_input_sets[i][1] / 20.0;
        test_input_sets[i][2] = denormalized_test_input_sets[i][2] / 300.0;
        test_input_sets[i][3] = (denormalized_test_input_sets[i][3] - 80.0) / (120.0 - 80.0);
        test_input_sets[i][4] = (denormalized_test_input_sets[i][4] - 1.79) / (1.89 - 1.79);
    }
    //--------------------------------------------------------------------------------------

    float t0;
    float denormalized_test_output[testing_set_size];

    fp = fopen(test_output_file,"r");
    if(fp == NULL)
        printf("Failed to read %s\nerrno: %d\n", test_output_file, errno);
    for(int i = 0; i < testing_set_size; i++){
        fscanf(fp, "%f", &t0);
        denormalized_test_output[i] = t0;
    }
    fclose(fp);

    calc_t test_output[testing_set_size];
    for(int i = 0; i < testing_set_size; i++){
        test_output[i] = normalize((calc_t)(denormalized_test_output[i]), data_min, data_max);
    }

    calc_t output[training_set_size];
    for(int i = 0; i < training_set_size; i++){
        output[i] = normalize(denormalized_output[i], data_min, data_max);
    }

    fp = fopen(error_results_file,"w");
    if(fp == NULL)
        printf("Failed to write %s\nerrno: %d\n", error_results_file, errno);

    //train the network
    int i;
    struct timeval t1, t2;
    //fprintf(fp,"Iterations\t\t\tError\n");
    gettimeofday(&t1, NULL);
    calc_t error = 1.0;
    calc_t error_1 = 0.0;
    for(i = 0; i < training_iterations && error > target_error; i++) {
        error = 0.0;
        //fprintf(fp,"%d: ",i);
        for(int j = 0; j < training_set_size; j++) {
            error_1 = train_network(nn, input_sets[j], output[j]);      //train_network() can be replaced w/individual function calls
            //fprintf(fp,"%d\t%.2e\n",i*training_set_size+j,error_1);
            error += error_1;
        }
        error = error/training_set_size;
        fprintf(fp,"%d\t%.2e\n",i,error);
        //fprintf(fp,"%.2e\n",error);
    }
    gettimeofday(&t2, NULL);
    fclose(fp);
    printf("execution time(s): %.4f\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)*1.0E-6);
    printf("total error: %e\n", error);
    printf("total runs = %d\n", i);

    fp = fopen(test_results_file,"w");
    if(fp == NULL)
        printf("Failed to write %s\nerrno: %d", test_results_file, errno);

    calc_t difference;
    calc_t test_error = 0;

     //test network
    for(i = 0; i < testing_set_size; i++){
        feed_forward(nn, test_input_sets[i]);
        calc_t out = nn->layer[num_layers - 1]->node[0]->o;
        fprintf(fp,"%.2f, %.2f, ", denormalize(out, data_min, data_max), denormalized_test_output[i]);
        difference = denormalized_test_output[i] - denormalize(out, data_min, data_max);
        fprintf(fp,"Error = %.2f%\n", difference);
        test_error += fabs(difference);
    }
    fclose(fp);
    printf("Test Error: %e\n", test_error/testing_set_size);

    //dealloc neural network
    //dealloc(nn);
    return 0;
}

