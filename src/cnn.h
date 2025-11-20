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

// Print a short summary of the model weights (first kshow kernel values and
// first dshow dense weights) to stdout. Returns 0 on success.
int cnn_print_summary(TinyCNN *net, int kshow, int dshow);

/* If non-zero, the network will emit extra debug prints (activations). */
extern int cnn_debug;

// L2 regularization coefficient (modifiable at runtime)
extern float cnn_l2;

// Lightweight diagnostics: retrieve bias and mean weight of final dense layer
void cnn_get_dense2_stats(TinyCNN *net, float *bias_out, float *mean_w_out);

// Run a numeric gradient check on the network for a single image/label.
// Prints analytic vs numeric gradients for a few parameters and returns 0.
int cnn_gradcheck(TinyCNN *net, const Image *im, int label, float eps);

#endif // CNN_H
