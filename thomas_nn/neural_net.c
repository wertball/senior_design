#include "neural_net.h"
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>

Neural_Net* init_neural_net(Neural_Net_Init_Params *nnip){
    //temp variables
    int i, j;
    Layer *pl, *l, *nl;  //pointers to previous,current, and next layers
    Node *n;
    //seed-random number generator for initial weight randomization
    srand(time(NULL));

    //allocate memory for nn
    Neural_Net *nn = malloc(sizeof(*nn));
    //initialize network characteristics
    nn->l = nnip->l;        //set total # layers
    nn->nn = 0;             //set number of nodes to 0 as prep for running sum
    nn->nw = 0;             //set number of weights to 0 as prep for running sum
    nn->lc  = nnip->lc;     //set learning constant
    nn->cte = 1.0;          //set current total error
    //nn->data_min = nnip->data_min;
    //nn->data_max = nnip->data_max;
    //nn->data_range = nnip->data_range;
    //allocate layer pointer array memory
    nn->layer = malloc(sizeof(nn->layer) * nn->l);

    //allocate memory for layers and nodes and set number of node values
    for(i = 0; i < nn->l; i++) {
        l = malloc(sizeof(*l));                         //allocate memory for new layer
        l->nn = nnip->dim[i];                           //set number of nodes in layer
        l->node = malloc(sizeof(*(l->node)) * l->nn);   //set node pointer array for layer
        for(j = 0; j < l->nn; j++){
            n  = malloc(sizeof(*n));                //create node in layer
            l->node[j] = n;                         //update layer with new node
        }
        nn->layer[i] = l;                           //add layer to neural net
    }

    //-------------------------------input layer--------------------------------
    //layer parameters
    l = nn->layer[0];                   //input layer address
    nl = nn->layer[1];                  //next layer address
    l->nw = l->nn * nnip->dim[1];       //nw = nodes in this layer * nodes in next layer
    l->next = nl;                       //next layer is a hidden layer
    l->prev = NULL;                     //there is no previous layer for input layer
    //nn parameters
    nn->nw += l->nw;    //increment number of weights in neural net
    nn->nn += l->nn;    //increment number of nodes in neural net

    //node parameters
    for(i = 0; i < l->nn; i++){
        n = l->node[i];                 //current node base address
        n->nw = nnip->dim[1];           //set number of weights to number of nodes in next layer
        n->o = 0;                       //set output of node to 0
        n->d = 0;                       //set delta of node to 0
        n->cw = init_weights(n->nw, 4); //cw = nw size array of random weights
        n->uw = n->cw;                  //uw = cw
    }

    //-------------------------------hidden layers-------------------------------
    for(i = 1; i < nn->l - 1; i++) {
        //layer parameters
        pl = nn->layer[i - 1];              //previous layer address
        l = nn->layer[i];                   //input layer address
        nl = nn->layer[i + 1];              //next layer address
        l->nw = l->nn * nnip->dim[i + 1];   //nw = nodes in this layer * nodes in next layer
        l->next = nl;                       //next layer is a (hidden|output) layer
        l->prev = pl;                       //previous layer is a (hidden|input) layer
        //nn parameters
        nn->nw += l->nw;    //increment number of weights in neural net
        nn->nn += l->nn;    //increment number of nodes in neural net
        //node parameters
        for (j = 0; j < l->nn; j++) {
            n = l->node[j];                    //current node base address
            n->nw = nnip->dim[i + 1];          //set number of weights to number of nodes in next layer
            n->o = 1;                          //set output of node to 0
            n->d = 3;                          //set delta of node to 0
            n->cw = init_weights(n->nw, 4);    //cw = nw size array of random weights
            n->uw = n->cw;                     //uw = cw
        }
    }

    //-------------------------------output layer--------------------------------
    //layer parameters
    pl = nn->layer[nn->l-2];            //previous layer address
    l = nn->layer[nn->l-1];             //input layer address
    l->nw = 0;                          //there are zero weights in output layer
    l->next = NULL;                     //there is no next layer
    l->prev = pl;                       //previous layer is a hidden layer
    //nn parameters
    nn->nn += l->nn;    //increment number of nodes in neural net

    //node parameters
    for(j = 0; j < l->nn; j++){
        n = l->node[j];                 //current node base address
        n->nw = 0;                      //set number of weights to 0
        n->o = 0;                       //set output of node to 0
        n->d = 0;                       //set delta of node to 0
        n->cw = NULL;                   //cw = NULL pointer
        n->uw = n->cw;                  //uw = cw
    }
    return nn;
}

calc_t* init_weights(int size, calc_t bound){
    //allocate memory for weight array
    calc_t* arr = malloc(sizeof(*arr) * size);
    //randomize array with values between range [0, upper bound]
    for(int i = 0; i < size; i++){
        calc_t sum = 0;
        if(rand() >= RAND_MAX / 2)
            sum += (calc_t)rand()/(calc_t)(RAND_MAX/bound);
        else
            sum -= (calc_t)rand()/(calc_t)(RAND_MAX/bound);
        arr[i] = sum;
    }
    return arr;
}

calc_t train_network(Neural_Net *nn, calc_t *input, calc_t output){
    //(1) feed forward
    feed_forward(nn, input);
    //(2) calculate total error output
    nn->cte = find_total_error(output, nn->layer[nn->l-1]->node[0]->o);
    //(3) backpropagate
    backpropagate(nn, output);
    //(4) assign new weights
    update_weights(nn);

    return nn->cte;
}

void feed_forward(Neural_Net *nn, calc_t *input){
    int i, j, k;    //iterators
    calc_t sum;     //running sum for node input MACs
    Layer *pl, *l;  //previous, current layer pointers

    //-----------------------input layer----------------------
    l = nn->layer[0];
	#pragma omp parallel for private(i)
    for(i = 0; i < l->nn; i++){
        l->node[i]->o = input[i];
    }

    //-----------------------hidden layers----------------------
    for(i = 1; i < (nn->l - 1); i++){
        l = nn->layer[i];
        pl = l->prev;
		#pragma omp parallel for private(j,k,sum)
        for(j = 0; j < l->nn; j++){
            for(sum = 0, k = 0; k < pl->nn; k++)
                sum +=  pl->node[k]->o * pl->node[k]->cw[j];
            l->node[j]->o = activate(sum);
        }
    }

    //-----------------------output layer----------------------
    l = nn->layer[nn->l-1];
    pl = l->prev;
	#pragma omp parallel for private(j,k,sum)
    for(j = 0; j < l->nn; j++){
        for(sum = 0, k = 0; k < pl->nn; k++)
            sum +=  pl->node[k]->o * pl->node[k]->cw[j];
        l->node[j]->o = normalize(sum, 0.549361, 3.43917);
    }
}

calc_t find_total_error(calc_t desired, calc_t actual){
    return (calc_t).5 * (desired - actual) * (desired - actual);
}

void backpropagate(Neural_Net *nn, calc_t output) {
    calc_t dw,          //dE/dw,
           delta,       //dE/dni
           lc = nn->lc; //learning constant
    int i, j;   //iterators
    Layer *pl,  //previous layer pointer
          *cl,  //current layer pointer
          *nl;  //next layer pointer

    //-------------------------output layer------------------------------
    cl = nn->layer[nn->l - 1];      //current layer = output layer
    pl = cl->prev;                  //previous layer = current layer -> prev
    for (i = 0; i < cl->nn; i++) {  //for every node in output layer
        //determine delta
        delta = (cl->node[i]->o - output) * (1); //delta = (dE/dno)(dno/dni)
        cl->node[i]->d = delta;
        //update previous layer nodes
        #pragma omp parallel for private(j,dw)
        for (j = 0; j < pl->nn; j++) {              //for every node in previous layer
            Node *pln = pl->node[j];
            dw = delta * pln->o;                    //dw = (delta)(node output)
            pln->uw[i] = pln->cw[i] - lc * dw;      //formula for new update weight
        }
    }

    //-------------------------remaining layers------------------------------
    cl = nn->layer[nn->l - 2];          //current layer = last hidden layer
    pl = cl->prev;                      //previous layer = current layer -> prev
    nl = cl->next;                      //next layer = current layer -> next
    while (pl != NULL) {                //iterating through layers until cl is input layer
        for (i = 0; i < cl->nn; i++) {      //for every node in current layer
            Node *cln = cl->node[i];        //current layer node
            //determine delta
            delta = 0.0;                    //initialize delta for running sum
            for (j = 0; j < nl->nn; j++) {            //for every node in next layer
                Node *nln = nl->node[j];
                //sub-delta = (dE/di)(di/do)(do/di),
                delta += (nln->d) * (cln->cw[j]) * (cln->o * (1 - cln->o));
            }
            cln->d = delta;
            //update previous layer nodes
            for (j = 0; j < pl->nn; j++) {          //for every node in previous layer
                Node *pln = pl->node[j];
                dw = delta * pln->o;                //dw = (delta)(node output)
                pln->uw[i] = pln->cw[i] - lc * dw;  //formula for new update weight
            }
        }
        nl = cl;        //next layer = current layer -> next
        cl = pl;        //current layer = output layer
        pl = pl->prev;  //previous layer = current layer -> prev
    }
}

void update_weights(Neural_Net *nn){
    for(Layer* l = nn->layer[0]; l != NULL; l = l->next){
        for(int i = 0; i < l->nn; i++){
            Node* n = l->node[i];
            n->cw = n->uw;
        }
    }
}

calc_t activate(calc_t x){
    return (calc_t)1.0 / ((calc_t)1.0 + (calc_t) exp(-x));
}

calc_t percent_error(calc_t desired, calc_t actual){
    return fabs((actual - desired) / desired * (calc_t)100.0);
}

calc_t normalize(calc_t input, calc_t min, calc_t max){
    return (input - min) / (max - min);
}

calc_t denormalize(calc_t input, calc_t min, calc_t max){
    return (max - min) * input + min;
}

//help functions
void printWeights(Neural_Net *nn){
    printf("Listing Weights ----------------------------------------------------------------\n");
    for(int i = 0; i < nn->l; i++){
        printf("Layer[%d]:\n", i);
        for(int j = 0; j < nn->layer[i]->nn; j++){
            printf("\tNode[%d]:\n", j);
            for(int k = 0; k < nn->layer[i]->node[j]->nw; k++){
                printf("\t\tWeight[%d]: %f\n", k, nn->layer[i]->node[j]->cw[k]);
            }
        }
    }
    printf("--------------------------------------------------------------------------------\n");
}
