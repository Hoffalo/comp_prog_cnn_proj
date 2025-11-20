#include "image.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <errno.h>

#if defined(HAVE_STB_IMAGE)
// If the user enabled USE_STB in CMake and provided third_party/stb_image.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

Image *image_create(int w, int h) {
    Image *im = malloc(sizeof(Image));
    if (!im) return NULL;
    im->w = w; im->h = h;
    im->data = calloc(w * h, sizeof(float));
    if (!im->data) { free(im); return NULL; }
    return im;
}

void image_free(Image *im) {
    if (!im) return;
    free(im->data);
    free(im);
}

// Circle for cat, diagonal stripe for dog. Values in [0,1].
Image *generate_synthetic(int w, int h, int cat_label) {
    Image *im = image_create(w, h);
    if (!im) return NULL;
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    float r = fminf(w,h) * 0.28f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float val = 0.0f;
            if (cat_label) {
                float dx = x - cx; float dy = y - cy;
                float d = sqrtf(dx*dx + dy*dy);
                val = d < r ? 1.0f : 0.0f;
                // add slight blur
                if (d < r + 1.5f && d > r - 1.5f) val = 0.6f;
            } else {
                // diagonal stripes
                int stripe = ((x + y) / 3) % 2;
                val = stripe ? 1.0f : 0.0f;
            }
            im->data[y * w + x] = val;
        }
    }
    return im;
}

static Image *image_resize_nn(const Image *src, int tw, int th) {
    Image *dst = image_create(tw, th);
    if (!dst) return NULL;
    for (int y = 0; y < th; ++y) {
        for (int x = 0; x < tw; ++x) {
            int sx = x * src->w / tw;
            int sy = y * src->h / th;
            dst->data[y * tw + x] = src->data[sy * src->w + sx];
        }
    }
    return dst;
}

// Simple PGM (P5/P2) loader fallback: returns grayscale float image in [0,1]
static Image *load_pgm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char header[3] = {0};
    if (fscanf(f, "%2s", header) != 1) { fclose(f); return NULL; }
    int is_binary = 0;
    if (strcmp(header, "P5") == 0) is_binary = 1;
    else if (strcmp(header, "P2") == 0) is_binary = 0;
    else { fclose(f); return NULL; }
    // skip comments and read width height maxval
    int c = fgetc(f);
    while (isspace(c)) c = fgetc(f);
    while (c == '#') { // skip comment line
        while (c != '\n' && c != EOF) c = fgetc(f);
        c = fgetc(f);
    }
    ungetc(c, f);
    int w,h,mv;
    if (fscanf(f, "%d %d %d", &w, &h, &mv) != 3) { fclose(f); return NULL; }
    // consume single whitespace before pixel data
    fgetc(f);
    Image *im = image_create(w,h);
    if (!im) { fclose(f); return NULL; }
    if (is_binary) {
        for (int i = 0; i < w*h; ++i) {
            int v = fgetc(f);
            if (v == EOF) v = 0;
            im->data[i] = (float)v / (float)mv;
        }
    } else {
        for (int i = 0; i < w*h; ++i) {
            int v = 0; if (fscanf(f, "%d", &v) != 1) v = 0;
            im->data[i] = (float)v / (float)mv;
        }
    }
    fclose(f);
    return im;
}

Image *image_load_file(const char *path, int target_w, int target_h) {
#if defined(HAVE_STB_IMAGE)
    int w,h,channels;
    unsigned char *data = stbi_load(path, &w, &h, &channels, 0);
    if (!data) return NULL;
    Image *tmp = image_create(w,h);
    if (!tmp) { stbi_image_free(data); return NULL; }
    // convert to grayscale
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        int idx = (y*w + x) * channels;
        float v = 0.0f;
        if (channels == 1) v = data[y*w + x] / 255.0f;
        else {
            float r = data[idx+0] / 255.0f;
            float g = data[idx+1] / 255.0f;
            float b = data[idx+2] / 255.0f;
            v = 0.2126f*r + 0.7152f*g + 0.0722f*b;
        }
        tmp->data[y*w + x] = v;
    }
    stbi_image_free(data);
    // Normalize to zero mean/unit variance
    if (w == target_w && h == target_h) {
        float sum = 0.0f, sum2 = 0.0f;
        int N = w*h;
        for (int i = 0; i < N; ++i) { sum += tmp->data[i]; sum2 += tmp->data[i]*tmp->data[i]; }
        float mean = sum / N;
        float var = sum2 / N - mean*mean;
        float std = sqrtf(var + 1e-6f);
        for (int i = 0; i < N; ++i) tmp->data[i] = (tmp->data[i] - mean) / std;
        return tmp;
    }
    Image *res = image_resize_nn(tmp, target_w, target_h);
    // Normalize resized image
    float sum = 0.0f, sum2 = 0.0f;
    int N = target_w*target_h;
    for (int i = 0; i < N; ++i) { sum += res->data[i]; sum2 += res->data[i]*res->data[i]; }
    float mean = sum / N;
    float var = sum2 / N - mean*mean;
    float std = sqrtf(var + 1e-6f);
    for (int i = 0; i < N; ++i) res->data[i] = (res->data[i] - mean) / std;
    image_free(tmp);
    return res;
#else
    // fallback: try PGM loader
    Image *im = load_pgm(path);
    if (!im) return NULL;
    if (im->w == target_w && im->h == target_h) return im;
    Image *res = image_resize_nn(im, target_w, target_h);
    image_free(im);
    return res;
#endif
}
