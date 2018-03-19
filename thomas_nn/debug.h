#ifndef DEBUG_HEADER
#define DEBUG_HEADER

#include <stdio.h>
#include "neural_net.h"

#define DEBUGGING //comment this flag to disable debug messages

//debug functions
void printWeights(Neural_Net *nn);
void printInputs(Input_Set_Params *isp, calc_t** input);
void printOutputs(Input_Set_Params *isp, calc_t* output);

#ifdef DEBUGGING
# define DEBUG_PRINT(x) printf x
# define DEBUG_PRINT_WEIGHTS(x) printWeights x
# define DEBUG_PRINT_INPUTS(x) printInputs x
# define DEBUG_PRINT_OUTPUTS(x) printOutputs x
#else
# define DEBUG_PRINT(x) do {} while (0)
# define DEBUG_PRINT_WEIGHTS(x) do {} while (0)
# define DEBUG_PRINT_INPUTS(x) do {} while (0)
# define DEBUG_PRINT_OUTPUTS(x) do {} while (0)
#endif


#endif

