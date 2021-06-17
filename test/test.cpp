#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ndarray.hpp"
#include "network.hpp"

char** names;
ndarray_t* params;

typedef struct {
    int value;
} int_t;

int_t* addsub(int_t** diff, int x, int y) {
    int_t* sum = (int_t*)malloc(sizeof(int_t));
    *diff = (int_t*)malloc(sizeof(int_t));
    
    sum->value = x + y;
    (*diff)->value = x - y;
    
    return sum;
}

int main() {
    int_t* sum;
    int_t* diff;
    sum = addsub(&diff, 5, 4);
    
    printf("sum: %d\n", sum->value);
    printf("diff: %d\n", diff->value);
    return 0;
}
