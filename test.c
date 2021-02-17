#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ndarray.h"
#include "network.h"

char** names;
ndarray_t* params;

int main() {
    char str[100];
    sprintf(str,"%s%s%d\n","hoge","piyo",2);
    printf("%s", str);
    sprintf(str,"%s%s%d\n","fuga","buma",4);
    printf("%s", str);
    return 0;
}

