#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ndarray.h"
#include "network.h"

char** names;
ndarray_t* params;

int main() {
    printf("loading params...\n");
    load_params(&names, &params);
    printf("\tdone.\n");
    
    float Max = params[0].data[0];
    float Min = params[0].data[0];
    float Absmin = 100;
    for (int i = 0; i < 213; i++) {
        //printf("%s\n", names[i]);
        float max = params[i].data[0];
        float min = params[i].data[0];
        float absmin = 100;
        for (int c = 1; c < params[i].length; c++) {
            float e = params[i].data[c];
            if (e > max) max = e;
            if (e < min) min = e;
            if (fabs(e) > 0 && fabs(e) < absmin) absmin = fabs(e);
            
            if (e > Max) Max = e;
            if (e < Min) Min = e;
            if (fabs(e) > 0 && fabs(e) < Absmin) Absmin = fabs(e);
        }
        //printf("\tmax: %f\n", max);
        //printf("\tmin: %f\n", min);
        //printf("%e\n", absmin);
    }
    printf("%e\n", Absmin);
            
    return 0;
}

