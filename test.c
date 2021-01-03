#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void hoge(char*** p) {
    *p = (char**)malloc(4 * sizeof(char*));
    for (int i = 0; i < 4; i++) {
        (*p)[i] = (char*)malloc(100 * sizeof(char));
    }
}

void piyo(int** num_p) {
    *num_p = (int*)malloc(4 * sizeof(int));
    for (int i = 0; i < 4; i++) {
        (*num_p)[i] = i;
    }
}

int main() {
    return 0;
}

