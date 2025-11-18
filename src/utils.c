#include "utils.h"
#include <stdlib.h>
#include <math.h>

static unsigned int g_seed = 123456;

void rand_seed(unsigned int s) { g_seed = s; }

float frandf() {
    g_seed = 1664525u * g_seed + 1013904223u;
    return (float)(g_seed & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

float dsigmoidf_from_output(float y) { return y * (1.0f - y); }

void he_init(float *w, size_t n) {
    for (size_t i = 0; i < n; ++i) w[i] = frandf() * 0.1f;
}
