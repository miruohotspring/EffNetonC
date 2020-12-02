#include <stdio.h>
#include <stdlib.h>
#include "ndarray.h"
#include "network.h"

int main(int argc, char* argv[]) {
    printf("loading params...");
    ndarray_t* params = load_params();
    //printf("%f", params[3].data[251]);
    return 0;
}
