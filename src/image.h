#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

// Simple grayscale image container
typedef struct {
    int w, h;
    float *data; // row-major, values in [0,1]
} Image;

Image *image_create(int w, int h);
void image_free(Image *im);

// Generate synthetic "cat" (circle) or "dog" (diagonal stripe) pattern
Image *generate_synthetic(int w, int h, int cat_label);

// Load image from file and convert/resize to grayscale float image in [0,1].
// If the build was configured with stb support, this will use stb_image. Otherwise
// a simple PGM loader is available as a fallback (supports ASCII and binary P5/P2 PGM).
Image *image_load_file(const char *path, int target_w, int target_h);

#endif // IMAGE_H
