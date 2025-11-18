#ifndef CNN_H
#define CNN_H

#include "image.h"

// A tiny CNN with one conv layer -> relu -> maxpool -> dense -> sigmoid

typedef struct TinyCNN TinyCNN;

TinyCNN *cnn_create(int in_w, int in_h, int filters, int ksize, int pool);
void cnn_free(TinyCNN *net);

// Forward: returns output probability in [0,1]
float cnn_forward(TinyCNN *net, const Image *im);

// Backward step using label (0 or 1) and learning rate
float cnn_backward_and_update(TinyCNN *net, const Image *im, int label, float lr);

// Save/load (very simple)
int cnn_save(TinyCNN *net, const char *path);
int cnn_load(TinyCNN *net, const char *path);

#endif // CNN_H
