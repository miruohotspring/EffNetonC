#include <stdio.h>
#include <stdlib.h>
#include "ndarray.h"
#include "network.h"

int main(int argc, char* argv[]) {
    char** names;
    ndarray_t* params;
    printf("loading params...");
    load_params(&names, &params);
    
    printf("%s\n", names[3]);
    printf("%f\n", params[3].data[0]);
    return 0;
}
