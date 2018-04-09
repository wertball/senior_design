#include "neural_net.h"
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdio.h>

Neural_Net* init_neural_net(Neural_Net_Init_Params *nnip){
    //temp variables
    int i, j, k;
    Layer *pl, *l, *nl;  //pointers to previous,current, and next layers
    Node *n;
    Thread_Data *thread;
    //seed-random number generator for initial weight randomization
    srand(time(NULL));

    //allocate memory for nn
    Neural_Net *nn = malloc(sizeof(*nn));
    //initialize network characteristics
    nn->l = nnip->l;            //set total # layers
    nn->nt = nnip->nthreads;    //set total # threads this nn is intended to be executed with
    nn->nn = 0;                 //set number of nodes to 0 as prep for running sum
    nn->nw = 0;                 //set number of weights to 0 as prep for running sum
    nn->lc  = nnip->lc;         //set learning constant
    nn->cte = 1.0;              //set current total error
    //allocate layer pointer array memory
    nn->layer = malloc(sizeof(nn->layer) * nn->l);

    //allocate memory for layers and nodes and set number of node values
    for(i = 0; i < nn->l; i++) {
        l = malloc(sizeof(*l));                        //allocate memory for new layer
        l->nn = nnip->dim[i];                          //set number of nodes in layer
        l->node = malloc(sizeof(*(l->node)) * l->nn);  //set node pointer array for layer
        for(j = 0; j < l->nn; j++){
            n = malloc(sizeof(*n));                         //create node in layer
            n->thread = malloc(sizeof(*thread) * nn->nt);   //malloc thread data ptr array
            for(k = 0; k < nn->nt; k++){
                thread = malloc(sizeof(*thread));           //malloc thread data
                n->thread[k] = thread;                      //update node with new thread data
            }
            l->node[j] = n;                                 //update layer with new node
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
        n->cw = init_weights(n->nw, 1); //cw = nw size array of random weights

        //initialize thread data
        for(k = 0; k < nn->nt; k++) {
            thread = n->thread[k];
            thread->uw = init_weights(n->nw, 0); //set update weights to 0
            thread->o = 0;                       //set output of node to 0
            thread->d = 0;                       //set delta of node to 0
        }
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
            n->cw = init_weights(n->nw, 1);    //cw = nw size array of random weights

            //initialize thread data
            for(k = 0; k < nn->nt; k++) {
                thread = n->thread[k];
                thread->uw = init_weights(n->nw, 0); //set update weights to 0
                thread->o = 0;                       //set output of node to 0
                thread->d = 0;                       //set delta of node to 0
            }
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
        n->cw = NULL;                   //cw = NULL pointer

        //initialize thread data
        for(k = 0; k < nn->nt; k++) {
            thread = n->thread[k];
            thread->uw = NULL;      //set update weights to 0
            thread->o = 0;          //set output of node to 0
            thread->d = 0;          //set delta of node to 0
        }
    }
    return nn;
}

calc_t* init_weights(int size, calc_t bound){
    //allocate memory for weight array
    calc_t* arr = malloc(sizeof(*arr) * size);
    //randomize array with values between range [0, upper bound]
    for(int i = 0; i < size; i++){
        calc_t sum = 0;
        if(bound){
            if (rand() >= RAND_MAX / 2)
                sum += (calc_t) rand() / (calc_t) (RAND_MAX / bound);
            else
                sum -= (calc_t) rand() / (calc_t) (RAND_MAX / bound);
        }
        arr[i] = sum;
    }
    return arr;
}

void feed_forward(Neural_Net *nn, calc_t *input, int tid, calc_t *data_range){
    int i, j, k;    //iterators
    calc_t sum;     //running sum for node input MACs
    Layer *pl, *l;  //previous, current layer pointers

    //-----------------------input layer----------------------
    l = nn->layer[0];
    for(i = 0; i < l->nn; i++){
        l->node[i]->thread[tid]->o = input[i];
    }

    //-----------------------hidden layers----------------------
    for(i = 1; i < (nn->l - 1); i++){
        l = nn->layer[i];
        pl = l->prev;
        for(j = 0; j < l->nn; j++){
            for(sum = 0, k = 0; k < pl->nn; k++)
                sum +=  pl->node[k]->thread[tid]->o * pl->node[k]->cw[j];
            l->node[j]->thread[tid]->o = activate(sum);
        }
    }

    //-----------------------output layer----------------------
    l = nn->layer[nn->l-1];
    pl = l->prev;
    for(j = 0; j < l->nn; j++){
        for(sum = 0, k = 0; k < pl->nn; k++)
            sum +=  pl->node[k]->thread[tid]->o * pl->node[k]->cw[j];
        l->node[j]->thread[tid]->o = normalize(sum, data_range);
    }
}

calc_t find_total_error(calc_t desired, calc_t actual){
    return (calc_t).5 * (desired - actual) * (desired - actual);
}

void backpropagate(Neural_Net *nn, calc_t output, int tid) {
    calc_t dw,           //dE/dw,
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
        delta = (cl->node[i]->thread[tid]->o - output) * (1); //delta = (dE/dno)(dno/dni)
        cl->node[i]->thread[tid]->d = delta;
        //update previous layer nodes
        for (j = 0; j < pl->nn; j++) {  //for every node in previous layer
            Node *pln = pl->node[j];
            dw = delta * pln->thread[tid]->o;        //dw = (delta)(node output)
            pln->thread[tid]->uw[i] += lc * dw;      //formula for new update weight
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
                delta += (nln->thread[tid]->d) * (cln->cw[j]) * (cln->thread[tid]->o * (1 - cln->thread[tid]->o));
            }
            cln->thread[tid]->d = delta;
            //update previous layer nodes
            for (j = 0; j < pl->nn; j++) {  //for every node in previous layer
                Node *pln = pl->node[j];
                dw = delta * pln->thread[tid]->o;        //dw = (delta)(node output)
                pln->thread[tid]->uw[i] += lc * dw;      //formula for new update weight
            }
        }
        nl = cl;        //next layer = current layer -> next
        cl = pl;        //current layer = output layer
        pl = pl->prev;  //previous layer = current layer -> prev
    }
}

void sync_update_weights(Neural_Net *nn, calc_t avg_divisor){
    calc_t sum;
    for(Layer* l = nn->layer[0]; l != nn->layer[nn->l-1]; l = l->next){
        for(int i = 0; i < l->nn; i++){
            Node* n = l->node[i];
            for(int j = 0; j < n->nw; j++) {
                sum = 0.0;
                Thread_Data** t = n->thread;
                for(int k = 0; k < nn->nt; k++) {
                    sum += t[k]->uw[j];
                    t[k]->uw[j] = 0.0;
                }
                n->cw[j] -= sum / avg_divisor;
            }
        }
    }
}

calc_t activate(calc_t x){
    return (calc_t)1.0 / ((calc_t)1.0 + (calc_t) exp(-x));
}

calc_t percent_error(calc_t desired, calc_t actual){
    return fabs((actual - desired) / desired * (calc_t)100.0);
}

calc_t normalize(calc_t input, calc_t *data_range){
    //index 0 => min
    //index 1 => max
    return (input - data_range[0]) / (data_range[1] - data_range[0]);
}

calc_t denormalize(calc_t input, calc_t *data_range){
    //index 0 => min
    //index 1 => max
    return input * (data_range[1] - data_range[0]) + data_range[0];
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

void printNetwork(Neural_Net *nn){
    printf("Listing Network Characteristics -------------------------------------------------\n");
    printf("l: %d\n", nn->l);
    printf("nt: %d\n", nn->nt);
    printf("nn: %d\n", nn->nn);
    printf("nw: %d\n", nn->nw);
    printf("lc: %.4f\n", nn->lc);
    printf("cte: %.4f\n", nn->cte);
    printf("**layer: %p\n", nn->layer);
    for(int i = 0; i < nn->l; i++){
        Layer *l = nn->layer[i];
        printf("\tlayer[%d]: %p\n", i, l);
        printf("\t\tnn: %d\n", l->nn);
        printf("\t\tnw: %d\n", l->nw);
        printf("\t\t*next: %p\n", l->next);
        printf("\t\t*prev: %p\n", l->prev);
        printf("\t\t**node: %p\n", l->node);
        for(int j = 0; j < nn->layer[i]->nn; j++){
            Node *n = l->node[j];
            printf("\t\tNode[%d]: %p\n", j, n);
            printf("\t\t\tnw: %d\n", n->nw);
            printf("\t\t\tcw: %p\n", n->cw);
            for(int k = 0; k < n->nw; k++){
                printf("\t\t\t\tcw[%d]: %f\n", k, n->cw[k]);
            }
            printf("\t\t\t**thread: %p\n", n->thread);
            for(int k = 0; k < nn->nt; k++){
                Thread_Data *t = n->thread[k];
                printf("\t\t\t\tthread[%d]: %p\n", k, n->thread[k]);
                printf("\t\t\t\t\to: %f\n", t->o);
                printf("\t\t\t\t\td: %f\n", t->d);
                printf("\t\t\t\t\tuw: %p\n", t->uw);
                for(int ii = 0; ii < n->nw; ii++){
                    printf("\t\t\t\t\t\tuw[%d]: %f\n", ii, t->uw[ii]);
                }
            }
        }
    }
    printf("---------------------------------------------------------------------------------\n");
}
