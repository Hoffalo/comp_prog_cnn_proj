#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

float frandf();
void rand_seed(unsigned int s);
float sigmoidf(float x);
float dsigmoidf_from_output(float y);
void he_init(float *w, size_t n);

#endif // UTILS_H
