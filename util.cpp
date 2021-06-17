#include "util.hpp"
#include <math.h>
#include <iostream>

int read_num = 0;
int write_num = 0;

int total_read_num = 0;
int total_write_num = 0;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

float readmem(const float* array, int i) {
    read_num++;
    total_read_num++;
    return array[i];
}

void writemem(float* array, int i, float data) {
    write_num++;
    total_write_num++;
    array[i] = data;
}

void print_readnum() {
    std::cout << "read count: " << read_num << std::endl;
    read_num = 0;
}

void print_writenum() {
    std::cout << "write count: " << write_num << std::endl;
    write_num = 0;
}

void print_total_readnum() {
    std::cout << "total read: " << total_read_num << std::endl;
}

void print_total_writenum() {
    std::cout << "total write: " << total_write_num << std::endl;
}
