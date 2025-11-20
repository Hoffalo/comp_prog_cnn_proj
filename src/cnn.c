#include "cnn.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*
 * New architecture: conv1 (depthwise per-channel) -> ReLU -> conv2 (depthwise)
 * -> ReLU -> maxpool -> dense (relu) -> dense -> sigmoid
 *
 * For simplicity the conv layers are implemented depthwise (each filter
 * processes its own spatial map). This keeps the forward/backward code
 * simple and still increases representational capacity.
 */

struct TinyCNN {
    int in_w, in_h;
    int filters;
    int ksize;
    int pool;

    // conv1
    int out1_w, out1_h;
    float *kernels1; // filters * ksize * ksize
    float *bias1;
    float *conv1_out; // filters * out1_w * out1_h

    // conv2 (depthwise)
    int out2_w, out2_h;
    float *kernels2; // filters * ksize * ksize
    float *bias2;
    float *conv2_out; // filters * out2_w * out2_h

    // pooling
    int *pool_idx; // indices for maxpool over conv2_out

    // dense params
    float *dense_w; // dense_size x flat
    float dense_b;
    int dense_size;
    float *dense2_w; // second dense to scalar
    float dense2_b;
};

/* Global debug flag: when set, forward will print extra activations. */
int cnn_debug = 0;
// default L2
float cnn_l2 = 0.0001f;

static void conv_forward_input(const float *in, int in_w, int in_h,
                               float *out, int out_w, int out_h,
                               const float *kernels, const float *bias,
                               int filters, int ksize) {
    // standard conv where 'in' is a single-channel image
    for (int f = 0; f < filters; ++f) {
        const float *k = kernels + f * (ksize*ksize);
        float b = bias ? bias[f] : 0.0f;
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float s = b;
                for (int ky = 0; ky < ksize; ++ky) {
                    for (int kx = 0; kx < ksize; ++kx) {
                        int ix = x + kx;
                        int iy = y + ky;
                        float v = in[iy * in_w + ix];
                        s += v * k[ky * ksize + kx];
                    }
                }
                out[(f * out_h + y) * out_w + x] = s;
            }
        }
    }
}

static void conv_forward_depthwise_maps(const float *in_maps, int in_w, int in_h,
                                       float *out, int out_w, int out_h,
                                       const float *kernels, const float *bias,
                                       int filters, int ksize) {
    // depthwise conv: input is filter-stacked maps (f, h, w)
    for (int f = 0; f < filters; ++f) {
        const float *k = kernels + f * (ksize*ksize);
        float b = bias ? bias[f] : 0.0f;
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float s = b;
                for (int ky = 0; ky < ksize; ++ky) {
                    for (int kx = 0; kx < ksize; ++kx) {
                        int ix = x + kx;
                        int iy = y + ky;
                        float v = in_maps[(f * in_h + iy) * in_w + ix];
                        s += v * k[ky * ksize + kx];
                    }
                }
                out[(f * out_h + y) * out_w + x] = s;
            }
        }
    }
}

static void relu_inplace(float *arr, int n) {
    for (int i = 0; i < n; ++i) if (arr[i] < 0) arr[i] = 0;
}

static void maxpool_forward(float *in, int in_w, int in_h, int filters,
                            float *out, int pool, int *idx_buf) {
    int out_w = in_w / pool;
    int out_h = in_h / pool;
    for (int f = 0; f < filters; ++f) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float best = -1e9f; int besti = -1;
                for (int py = 0; py < pool; ++py) {
                    for (int px = 0; px < pool; ++px) {
                        int ix = x*pool + px;
                        int iy = y*pool + py;
                        int idx = (f * in_h + iy) * in_w + ix;
                        float v = in[idx];
                        if (v > best) { best = v; besti = idx; }
                    }
                }
                int out_idx = (f * out_h + y) * out_w + x;
                out[out_idx] = best;
                idx_buf[out_idx] = besti;
            }
        }
    }
}

TinyCNN *cnn_create(int in_w, int in_h, int filters, int ksize, int pool) {
    TinyCNN *net = calloc(1, sizeof(TinyCNN));
    net->in_w = in_w; net->in_h = in_h;
    net->filters = filters; net->ksize = ksize; net->pool = pool;

    net->out1_w = in_w - ksize + 1;
    net->out1_h = in_h - ksize + 1;
    net->kernels1 = malloc(sizeof(float) * (size_t)filters * ksize * ksize);
    net->bias1 = malloc(sizeof(float) * filters);
    he_init(net->kernels1, (size_t)filters * ksize * ksize);
    for (int i = 0; i < filters; ++i) net->bias1[i] = 0.0f;
    net->conv1_out = malloc(sizeof(float) * (size_t)filters * net->out1_w * net->out1_h);

    // conv2 operates depthwise on conv1_out (same spatial dims)
    net->out2_w = net->out1_w - ksize + 1;
    net->out2_h = net->out1_h - ksize + 1;
    net->kernels2 = malloc(sizeof(float) * (size_t)filters * ksize * ksize);
    net->bias2 = malloc(sizeof(float) * filters);
    he_init(net->kernels2, (size_t)filters * ksize * ksize);
    for (int i = 0; i < filters; ++i) net->bias2[i] = 0.0f;
    net->conv2_out = malloc(sizeof(float) * (size_t)filters * net->out2_w * net->out2_h);

    size_t conv2_size = (size_t)filters * net->out2_w * net->out2_h;
    net->pool_idx = malloc(sizeof(int) * conv2_size);

    int pw = net->out2_w / pool;
    int ph = net->out2_h / pool;
    size_t flat = (size_t)filters * pw * ph;
    net->dense_size = 32;
    net->dense_w = malloc(sizeof(float) * flat * net->dense_size);
    he_init(net->dense_w, flat * net->dense_size);
    net->dense_b = 0.0f;
    net->dense2_w = malloc(sizeof(float) * net->dense_size);
    he_init(net->dense2_w, net->dense_size);
    net->dense2_b = 0.0f;
    return net;
}

void cnn_free(TinyCNN *net) {
    if (!net) return;
    free(net->kernels1);
    free(net->bias1);
    free(net->conv1_out);
    free(net->kernels2);
    free(net->bias2);
    free(net->conv2_out);
    free(net->pool_idx);
    free(net->dense_w);
    free(net->dense2_w);
    free(net);
}

float cnn_forward(TinyCNN *net, const Image *im) {
    // conv1 (depthwise over input channels effectively)
    conv_forward_input(im->data, net->in_w, net->in_h,
                       net->conv1_out, net->out1_w, net->out1_h,
                       net->kernels1, net->bias1, net->filters, net->ksize);
    int c1_n = net->filters * net->out1_w * net->out1_h;
    relu_inplace(net->conv1_out, c1_n);

    // conv2 (depthwise over conv1 maps)
    conv_forward_depthwise_maps(net->conv1_out, net->out1_w, net->out1_h,
                               net->conv2_out, net->out2_w, net->out2_h,
                               net->kernels2, net->bias2, net->filters, net->ksize);
    int c2_n = net->filters * net->out2_w * net->out2_h;
    relu_inplace(net->conv2_out, c2_n);

    // pool
    int pw = net->out2_w / net->pool;
    int ph = net->out2_h / net->pool;
    size_t flat = (size_t)net->filters * pw * ph;
    float *pooled = malloc(sizeof(float) * flat);
    maxpool_forward(net->conv2_out, net->out2_w, net->out2_h, net->filters, pooled, net->pool, net->pool_idx);

    // dense1
    float dense1[32];
    for (int j = 0; j < net->dense_size; ++j) {
        float s = net->dense_b;
        for (size_t i = 0; i < flat; ++i) s += pooled[i] * net->dense_w[j*flat + i];
        dense1[j] = s > 0 ? s : 0;
    }
    // debug (print only when enabled to avoid flooding logs)
    if (cnn_debug) {
        printf("Dense1 activations: ");
        for (int j = 0; j < 4; ++j) printf("%.3f ", dense1[j]);
        printf("...\n");
    }

    float s2 = net->dense2_b;
    for (int j = 0; j < net->dense_size; ++j) s2 += dense1[j] * net->dense2_w[j];
    free(pooled);
    return sigmoidf(s2);
}

// Simplified backward: propagate scalar error to pooled positions and then
// update kernels2 and kernels1 (depthwise) using those gradients. This is
// still a very approximate backward pass but is consistent with the tiny
// training machinery used elsewhere in the project.
float cnn_backward_and_update(TinyCNN *net, const Image *im, int label, float lr) {
    // Forward recompute (conv outputs already stored by cnn_forward called earlier)
    float out = cnn_forward(net, im);
    float loss = -(label ? logf(out + 1e-8f) : logf(1 - out + 1e-8f));
    float ds = out - (float)label; // dL/ds for pre-sigmoid

    int pw = net->out2_w / net->pool;
    int ph = net->out2_h / net->pool;
    size_t flat = (size_t)net->filters * pw * ph;
    float *pooled = malloc(sizeof(float) * flat);
    maxpool_forward(net->conv2_out, net->out2_w, net->out2_h, net->filters, pooled, net->pool, net->pool_idx);

    const float l2 = cnn_l2;
    const float clip = 5.0f;

    // Recompute dense1 pre-activations and activations (needed for correct grads)
    int D = net->dense_size;
    float *dense1_pre = malloc(sizeof(float) * D);
    float *dense1 = malloc(sizeof(float) * D);
    for (int j = 0; j < D; ++j) {
        float s = net->dense_b;
        for (size_t i = 0; i < flat; ++i) s += pooled[i] * net->dense_w[j*(size_t)flat + i];
        dense1_pre[j] = s;
        dense1[j] = s > 0 ? s : 0;
    }

    // Compute gradients for dense layers but do not apply updates until all grads are computed
    float *gdense2_w = malloc(sizeof(float) * D);
    float gdense2_b = 0.0f;
    float *ddense1_pre = malloc(sizeof(float) * D);
    // compute ddense1_pre using current dense2_w (old values)
    for (int j = 0; j < D; ++j) {
        float val = ds * net->dense2_w[j];
        if (dense1_pre[j] <= 0.0f) val = 0.0f; // ReLU'
        ddense1_pre[j] = val;
    }
    // grad for dense2 weights and bias (use dense1 activations)
    for (int j = 0; j < D; ++j) {
        float g = ds * dense1[j];
        g += l2 * net->dense2_w[j];
        if (g > clip) g = clip; if (g < -clip) g = -clip;
        gdense2_w[j] = g;
    }
    gdense2_b = ds + l2 * net->dense2_b;
    if (gdense2_b > clip) gdense2_b = clip; if (gdense2_b < -clip) gdense2_b = -clip;

    // gradients for dense_w and dense_b
    float *gdense_w = malloc(sizeof(float) * (size_t)D * flat);
    float gdense_b = 0.0f;
    for (int j = 0; j < D; ++j) {
        for (size_t i = 0; i < flat; ++i) {
            float g = ddense1_pre[j] * pooled[i];
            g += l2 * net->dense_w[j*(size_t)flat + i];
            if (g > clip) g = clip; if (g < -clip) g = -clip;
            gdense_w[j*(size_t)flat + i] = g;
        }
        gdense_b += ddense1_pre[j];
    }
    gdense_b += l2 * net->dense_b;
    if (gdense_b > clip) gdense_b = clip; if (gdense_b < -clip) gdense_b = -clip;

    // Compute gradient w.r.t pooled inputs using the old dense_w values
    float *dpooled = calloc(flat, sizeof(float));
    for (int j = 0; j < D; ++j) {
        for (size_t i = 0; i < flat; ++i) {
            dpooled[i] += ddense1_pre[j] * net->dense_w[j*(size_t)flat + i];
        }
    }

    // Apply dense updates now (after computing all gradients)
    for (int j = 0; j < D; ++j) {
        for (size_t i = 0; i < flat; ++i) {
            net->dense_w[j*(size_t)flat + i] -= lr * gdense_w[j*(size_t)flat + i];
        }
    }
    net->dense_b -= lr * gdense_b;
    for (int j = 0; j < D; ++j) net->dense2_w[j] -= lr * gdense2_w[j];
    net->dense2_b -= lr * gdense2_b;

    // Map pooled gradients back to conv2 positions via maxpool indices
    int conv2_size = net->filters * net->out2_w * net->out2_h;
    float *dconv2 = calloc(conv2_size, sizeof(float));
    for (size_t i = 0; i < flat; ++i) {
        int idx = net->pool_idx[i];
        if (idx >= 0 && idx < conv2_size) dconv2[idx] += dpooled[i];
    }

    // Backprop through ReLU on conv2
    for (int i = 0; i < conv2_size; ++i) if (net->conv2_out[i] <= 0) dconv2[i] = 0;

    // update kernels2 (using conv1_out as input) -- same as before but using dconv2
    int ksz = net->ksize * net->ksize;
    for (int f = 0; f < net->filters; ++f) {
        float *k2 = net->kernels2 + f * ksz;
        for (int ky = 0; ky < net->ksize; ++ky) {
            for (int kx = 0; kx < net->ksize; ++kx) {
                float g = 0.0f;
                for (int y = 0; y < net->out2_h; ++y) {
                    for (int x = 0; x < net->out2_w; ++x) {
                        int idx = (f * net->out2_h + y) * net->out2_w + x;
                        float grad = dconv2[idx];
                        if (grad == 0.0f) continue;
                        int ix = x + kx; int iy = y + ky;
                        float v = net->conv1_out[(f * net->out1_h + iy) * net->out1_w + ix];
                        g += v * grad;
                    }
                }
                g += l2 * k2[ky * net->ksize + kx];
                if (g > clip) g = clip;
                if (g < -clip) g = -clip;
                k2[ky * net->ksize + kx] -= lr * g;
            }
        }
        // bias2
        float gb2 = 0.0f;
        for (int y = 0; y < net->out2_h; ++y) for (int x = 0; x < net->out2_w; ++x) {
            int idx = (f * net->out2_h + y) * net->out2_w + x;
            gb2 += dconv2[idx];
        }
        gb2 += l2 * net->bias2[f];
        if (gb2 > clip) gb2 = clip; if (gb2 < -clip) gb2 = -clip;
        net->bias2[f] -= lr * gb2;
    }

    // propagate dconv2 back to conv1 (depthwise deconvolution with kernels2)
    int conv1_size = net->filters * net->out1_w * net->out1_h;
    float *dconv1 = calloc(conv1_size, sizeof(float));
    for (int f = 0; f < net->filters; ++f) {
        const float *k2 = net->kernels2 + f * ksz;
        for (int y = 0; y < net->out2_h; ++y) {
            for (int x = 0; x < net->out2_w; ++x) {
                int idx2 = (f * net->out2_h + y) * net->out2_w + x;
                float g = dconv2[idx2];
                if (g == 0.0f) continue;
                for (int ky = 0; ky < net->ksize; ++ky) {
                    for (int kx = 0; kx < net->ksize; ++kx) {
                        int ix = x + kx; int iy = y + ky;
                        int idx1 = (f * net->out1_h + iy) * net->out1_w + ix;
                        // kernel not flipped intentionally (symmetry for small k)
                        dconv1[idx1] += k2[ky * net->ksize + kx] * g;
                    }
                }
            }
        }
    }

    // backprop through ReLU on conv1
    for (int i = 0; i < conv1_size; ++i) if (net->conv1_out[i] <= 0) dconv1[i] = 0;

    // update kernels1 using dconv1 and original image input
    for (int f = 0; f < net->filters; ++f) {
        float *k1 = net->kernels1 + f * ksz;
        for (int ky = 0; ky < net->ksize; ++ky) {
            for (int kx = 0; kx < net->ksize; ++kx) {
                float g = 0.0f;
                for (int y = 0; y < net->out1_h; ++y) {
                    for (int x = 0; x < net->out1_w; ++x) {
                        int idx1 = (f * net->out1_h + y) * net->out1_w + x;
                        float grad = dconv1[idx1];
                        if (grad == 0.0f) continue;
                        int ix = x + kx; int iy = y + ky;
                        float v = im->data[iy * net->in_w + ix];
                        g += v * grad;
                    }
                }
                g += l2 * k1[ky * net->ksize + kx];
                if (g > clip) g = clip; if (g < -clip) g = -clip;
                k1[ky * net->ksize + kx] -= lr * g;
            }
        }
        // bias1
        float gb1 = 0.0f;
        for (int y = 0; y < net->out1_h; ++y) for (int x = 0; x < net->out1_w; ++x) {
            int idx1 = (f * net->out1_h + y) * net->out1_w + x;
            gb1 += dconv1[idx1];
        }
        gb1 += l2 * net->bias1[f];
        if (gb1 > clip) gb1 = clip; if (gb1 < -clip) gb1 = -clip;
        net->bias1[f] -= lr * gb1;
    }

    // free allocated temporaries
    free(dense1_pre); free(dense1); free(gdense2_w); free(ddense1_pre);
    free(gdense_w);
    free(dpooled); free(dconv2); free(dconv1); free(pooled);
    return loss;
}

int cnn_save(TinyCNN *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&net->in_w, sizeof(int), 1, f);
    fwrite(&net->in_h, sizeof(int), 1, f);
    fwrite(&net->filters, sizeof(int), 1, f);
    fwrite(&net->ksize, sizeof(int), 1, f);
    // kernels1
    int kcount = net->filters * net->ksize * net->ksize;
    fwrite(net->kernels1, sizeof(float), kcount, f);
    fwrite(net->bias1, sizeof(float), net->filters, f);
    // kernels2
    fwrite(net->kernels2, sizeof(float), kcount, f);
    fwrite(net->bias2, sizeof(float), net->filters, f);
    // dense weights
    int flat = net->filters * (net->out2_w / net->pool) * (net->out2_h / net->pool);
    fwrite(net->dense_w, sizeof(float), flat * net->dense_size, f);
    fwrite(&net->dense_b, sizeof(float), 1, f);
    fwrite(net->dense2_w, sizeof(float), net->dense_size, f);
    fwrite(&net->dense2_b, sizeof(float), 1, f);
    fclose(f);
    return 0;
}

int cnn_load(TinyCNN *net, const char *path) {
    if (!net || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int in_w=0, in_h=0, filters=0, ksize=0;
    if (fread(&in_w, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&in_h, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&filters, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&ksize, sizeof(int), 1, f) != 1) { fclose(f); return -1; }

    if (net->in_w != in_w || net->in_h != in_h || net->filters != filters || net->ksize != ksize) {
        fclose(f);
        return -1;
    }
    int kcount = filters * ksize * ksize;
    if (fread(net->kernels1, sizeof(float), kcount, f) != (size_t)kcount) { fclose(f); return -1; }
    if (fread(net->bias1, sizeof(float), filters, f) != (size_t)filters) { fclose(f); return -1; }
    if (fread(net->kernels2, sizeof(float), kcount, f) != (size_t)kcount) { fclose(f); return -1; }
    if (fread(net->bias2, sizeof(float), filters, f) != (size_t)filters) { fclose(f); return -1; }

    int flat = net->filters * (net->out2_w / net->pool) * (net->out2_h / net->pool);
    if (fread(net->dense_w, sizeof(float), flat * net->dense_size, f) != (size_t)(flat * net->dense_size)) { fclose(f); return -1; }
    if (fread(&net->dense_b, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
    if (fread(net->dense2_w, sizeof(float), net->dense_size, f) != (size_t)net->dense_size) { fclose(f); return -1; }
    if (fread(&net->dense2_b, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

int cnn_print_summary(TinyCNN *net, int kshow, int dshow) {
    if (!net) return -1;
    printf("Model summary:\n");
    printf(" in: %d x %d, filters=%d, ksize=%d, pool=%d\n", net->in_w, net->in_h, net->filters, net->ksize, net->pool);
    int kcount = net->filters * net->ksize * net->ksize;
    int kprint = kshow < kcount ? kshow : kcount;
    printf(" first %d kernel1 values:\n", kprint);
    for (int i = 0; i < kprint; ++i) printf("  k1[%d]=%.6f\n", i, net->kernels1[i]);
    printf(" first %d kernel2 values:\n", kprint);
    for (int i = 0; i < kprint; ++i) printf("  k2[%d]=%.6f\n", i, net->kernels2[i]);
    printf(" bias1/bias2 sample:\n"); for (int i = 0; i < net->filters && i < 4; ++i) printf("  b1[%d]=%.6f b2[%d]=%.6f\n", i, net->bias1[i], i, net->bias2[i]);
    int flat = net->filters * (net->out2_w / net->pool) * (net->out2_h / net->pool);
    int dprint = dshow < flat ? dshow : flat;
    printf(" dense_b=%.6f\n", net->dense_b);
    printf(" first %d dense weights:\n", dprint);
    for (int i = 0; i < dprint; ++i) printf("  dw[%d]=%.6f\n", i, net->dense_w[i]);
    return 0;
}

void cnn_get_dense2_stats(TinyCNN *net, float *bias_out, float *mean_w_out) {
    if (!net) {
        if (bias_out) *bias_out = 0.0f;
        if (mean_w_out) *mean_w_out = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (int j = 0; j < net->dense_size; ++j) sum += net->dense2_w[j];
    if (mean_w_out) *mean_w_out = sum / (float)net->dense_size;
    if (bias_out) *bias_out = net->dense2_b;
}

// Helper: compute BCE loss for current net on image (assumes forwardable)
static float cnn_loss_on_image(TinyCNN *net, const Image *im, int label) {
    float out = cnn_forward(net, im);
    if (label) return -logf(out + 1e-8f);
    else return -logf(1.0f - out + 1e-8f);
}

int cnn_gradcheck(TinyCNN *net, const Image *im, int label, float eps) {
    if (!net || !im) return -1;
    // Run forward to populate conv outputs
    float out = cnn_forward(net, im);
    float loss0 = label ? -logf(out + 1e-8f) : -logf(1.0f - out + 1e-8f);

    const float l2 = cnn_l2;

    // We'll compute analytic gradients for:
    // - dense_w[0,0]
    // - kernels1[0]
    int pw = net->out2_w / net->pool;
    int ph = net->out2_h / net->pool;
    size_t flat = (size_t)net->filters * pw * ph;
    int D = net->dense_size;

    // Recompute pooled and dense1 pre/act as in backward
    float *pooled = malloc(sizeof(float) * flat);
    maxpool_forward(net->conv2_out, net->out2_w, net->out2_h, net->filters, pooled, net->pool, net->pool_idx);
    float *dense1_pre = malloc(sizeof(float) * D);
    float *dense1 = malloc(sizeof(float) * D);
    for (int j = 0; j < D; ++j) {
        float s = net->dense_b;
        for (size_t i = 0; i < flat; ++i) s += pooled[i] * net->dense_w[j*(size_t)flat + i];
        dense1_pre[j] = s; dense1[j] = s > 0 ? s : 0;
    }

    // ds = out - label
    float ds = out - (float)label;

    // analytic gradient for dense2_w[j] = ds * dense1[j] + l2 * w
    float ana_dense2_w0 = ds * dense1[0] + l2 * net->dense2_w[0];

    // analytic gradient for dense_w[0,0] = ddense1_pre[0] * pooled[0] + l2 * w
    // ddense1_pre = ds * dense2_w[j] * ReLU'
    float dd1_0 = (dense1_pre[0] <= 0.0f) ? 0.0f : ds * net->dense2_w[0];
    float ana_dense_w_0_0 = dd1_0 * pooled[0] + l2 * net->dense_w[0*(size_t)flat + 0];

    // For kernel1 gradient analytic: we need dconv1 from conv2 backprop
    // We'll roughly follow backward: compute dpooled then map back to dconv2,
    // backprop through conv2 to dconv1, then compute gradient on kernels1.

    // compute dpooled using ddense1_pre and dense_w (old values)
    float *dpooled = calloc(flat, sizeof(float));
    for (int j = 0; j < D; ++j) {
        for (size_t i = 0; i < flat; ++i) dpooled[i] += ((dense1_pre[j] <= 0.0f) ? 0.0f : ds * net->dense2_w[j]) * net->dense_w[j*(size_t)flat + i];
    }
    int conv2_size = net->filters * net->out2_w * net->out2_h;
    float *dconv2 = calloc(conv2_size, sizeof(float));
    for (size_t i = 0; i < flat; ++i) {
        int idx = net->pool_idx[i];
        if (idx >= 0 && idx < conv2_size) dconv2[idx] += dpooled[i];
    }
    for (int i = 0; i < conv2_size; ++i) if (net->conv2_out[i] <= 0) dconv2[i] = 0;

    int ksz = net->ksize * net->ksize;
    int conv1_size = net->filters * net->out1_w * net->out1_h;
    float *dconv1 = calloc(conv1_size, sizeof(float));
    for (int f = 0; f < net->filters; ++f) {
        const float *k2 = net->kernels2 + f * ksz;
        for (int y = 0; y < net->out2_h; ++y) {
            for (int x = 0; x < net->out2_w; ++x) {
                int idx2 = (f * net->out2_h + y) * net->out2_w + x;
                float g = dconv2[idx2]; if (g == 0.0f) continue;
                for (int ky = 0; ky < net->ksize; ++ky) for (int kx = 0; kx < net->ksize; ++kx) {
                    int ix = x + kx; int iy = y + ky;
                    int idx1 = (f * net->out1_h + iy) * net->out1_w + ix;
                    dconv1[idx1] += k2[ky * net->ksize + kx] * g;
                }
            }
        }
    }
    for (int i = 0; i < conv1_size; ++i) if (net->conv1_out[i] <= 0) dconv1[i] = 0;

    // analytic gradient for kernel1[0,0] (first kernel element of filter 0)
    float ana_k1_0 = 0.0f;
    for (int y = 0; y < net->out1_h; ++y) for (int x = 0; x < net->out1_w; ++x) {
        int idx1 = (0 * net->out1_h + y) * net->out1_w + x;
        float grad = dconv1[idx1]; if (grad == 0.0f) continue;
        // kernel element corresponds to ky=0 kx=0 -> input pixel at (x+0,y+0)
        int ix = x + 0; int iy = y + 0;
        float v = im->data[iy * net->in_w + ix];
        ana_k1_0 += v * grad;
    }
    ana_k1_0 += l2 * net->kernels1[0];

    // Numeric gradients via central difference
    float num_dense2_w0, num_dense_w_0_0, num_k1_0;

    // dense2_w[0]
    net->dense2_w[0] += eps;
    float lp = cnn_loss_on_image(net, im, label);
    net->dense2_w[0] -= 2.0f * eps;
    float lm = cnn_loss_on_image(net, im, label);
    net->dense2_w[0] += eps; // restore
    num_dense2_w0 = (lp - lm) / (2.0f * eps);

    // dense_w[0,0]
    net->dense_w[0*(size_t)flat + 0] += eps;
    lp = cnn_loss_on_image(net, im, label);
    net->dense_w[0*(size_t)flat + 0] -= 2.0f * eps;
    lm = cnn_loss_on_image(net, im, label);
    net->dense_w[0*(size_t)flat + 0] += eps;
    num_dense_w_0_0 = (lp - lm) / (2.0f * eps);

    // kernels1[0]
    net->kernels1[0] += eps;
    lp = cnn_loss_on_image(net, im, label);
    net->kernels1[0] -= 2.0f * eps;
    lm = cnn_loss_on_image(net, im, label);
    net->kernels1[0] += eps;
    num_k1_0 = (lp - lm) / (2.0f * eps);

    // Print comparison
    float rel_err_dense2 = fabsf(ana_dense2_w0 - num_dense2_w0) / fmaxf(1e-8f, fabsf(ana_dense2_w0) + fabsf(num_dense2_w0));
    float rel_err_densew = fabsf(ana_dense_w_0_0 - num_dense_w_0_0) / fmaxf(1e-8f, fabsf(ana_dense_w_0_0) + fabsf(num_dense_w_0_0));
    float rel_err_k1 = fabsf(ana_k1_0 - num_k1_0) / fmaxf(1e-8f, fabsf(ana_k1_0) + fabsf(num_k1_0));

    printf("Gradcheck results (eps=%.6g):\n", eps);
    printf(" dense2_w[0]: analytic=%.8g numeric=%.8g rel_err=%.8g\n", ana_dense2_w0, num_dense2_w0, rel_err_dense2);
    printf(" dense_w[0,0]: analytic=%.8g numeric=%.8g rel_err=%.8g\n", ana_dense_w_0_0, num_dense_w_0_0, rel_err_densew);
    printf(" kernels1[0]: analytic=%.8g numeric=%.8g rel_err=%.8g\n", ana_k1_0, num_k1_0, rel_err_k1);

    // cleanup
    free(pooled); free(dense1_pre); free(dense1); free(dpooled); free(dconv2); free(dconv1);
    return 0;
}
