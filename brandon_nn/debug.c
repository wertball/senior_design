#include "debug.h"

void printWeights(Neural_Net *nn){
	DEBUG_PRINT(("Listing Weights ----------------------------------------------------------------\n"));
	for(int i = 0; i < nn->l; i++){
		DEBUG_PRINT(("Layer[%d]:\n", i));
		for(int j = 0; j < nn->layer[i]->nn; j++){
			DEBUG_PRINT(("\tNode[%d]:\n", j));
            for(int k = 0; k < nn->layer[i]->node[j]->nw; k++){
                DEBUG_PRINT(("\t\tWeight[%d]: %f\n", k, nn->layer[i]->node[j]->cw[k]));
            }
		}
	}
	DEBUG_PRINT(("--------------------------------------------------------------------------------\n"));
	return;
}

void printInputs(Input_Set_Params *isp, calc_t** input){
    DEBUG_PRINT(("Listing Inputs------------------------------------------------------------------\n"));
    for(int i = 0; i < isp->set_size; i++){
        DEBUG_PRINT(("Set %d: {", i));
        for(int j = 0; j < isp->input_set_size; j++){
            DEBUG_PRINT((" %f,", input[i][j]));
        }
        DEBUG_PRINT((" }\n"));
    }
    DEBUG_PRINT(("--------------------------------------------------------------------------------\n"));
}

void printOutputs(Input_Set_Params *isp, calc_t* output){
    DEBUG_PRINT(("Listing Outputs-----------------------------------------------------------------\n"));
    for(int i = 0; i < isp->set_size; i++){
        DEBUG_PRINT(("Output of Set %d: %f\n", i, output[i]));
    }
    DEBUG_PRINT(("--------------------------------------------------------------------------------\n"));
}