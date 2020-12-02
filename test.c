#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void hoge(int* size) {
    printf("%d\n", size[1]);
}

int main() {
    int size[6] = {0, 1, 2, 3, 4, 5};
    hoge(size);
    hoge(&size[0]);
    hoge(&size[3]);
    return 0;
}

