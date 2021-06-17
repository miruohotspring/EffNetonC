#ifndef UTIL_HPP
#define UTIL_HPP

float sigmoid(float x);
float readmem(const float* array, int i);
void writemem(float* array, int i, float data);
void print_readnum();
void print_writenum();
void print_total_readnum();
void print_total_writenum();

#endif