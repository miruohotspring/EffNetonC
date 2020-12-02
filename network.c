#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "ndarray.h"
#include "network.h"

ndarray_t* create_param_from_name(char* name) {
    ndarray_t* param = (ndarray_t*)malloc(sizeof(ndarray_t));
    int length = 1;
    int dim;
    int* size;
    double* data;
    
    FILE* f;
    char path[100] = "./data/";
    char c;
    char s[128];
    int j = 0;
    strcat(path, name);
    strcat(path, ".txt");
    if((f = fopen(path, "r")) == NULL) {
        printf("file open error");
        exit(EXIT_FAILURE);
    }
    if((dim = fgetc(f) - '0') == 0) return NULL;
    while((fgetc(f)) != ',');
    
    size = (int*)malloc(sizeof(int)*dim);
    for(int i = 0; i < dim; i++) {
        for(j = 0; (c = fgetc(f)), (c != ',' && c != EOF); j++) s[j] = c;
        s[j] = 0;
        size[i] = atoi(s);
        length *= size[i];
    }
    
    data = (double*)malloc(sizeof(double)*length);
    for(int i = 0; i < length; i++) {
        for(j = 0; (c = fgetc(f)), (c != ',' && c != EOF); j++) s[j] = c;
        s[j] = 0;
        data[i] = atof(s);
    }
    
    param->length = length;
    param->dim = dim;
    param->size = size;
    param->data = data;
    
    fclose(f);
    return param;
}

ndarray_t* load_params() {
    ndarray_t* params = (ndarray_t*)malloc(sizeof(ndarray_t)*213);
    
    FILE* f;
    char readline[100];
    
    if((f = fopen("./data/module_list.txt", "r")) == NULL) {
        printf("file open error: cannot open list");
        exit(EXIT_FAILURE);
    }
    int i = 0;
    while(fgets(readline, 100, f) != NULL) {
        readline[strcspn(readline, "\n")] = 0;
        printf("%s\n", readline);
        ndarray_t* param = create_param_from_name(readline);
        if (param == NULL) continue;
        params[i] = *param;
        i++;
    }    
    fclose(f);
    return params;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-1 * x));
}

void swish(ndarray_t* output, const ndarray_t* input) {
    assert(output->size == input->size);
    for (int i = 0; i < output->length; i++) {
        output->data[i] = sigmoid(input->data[i]);
    }
}





































