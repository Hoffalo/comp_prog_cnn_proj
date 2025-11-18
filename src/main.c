#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include "image.h"
#include "cnn.h"
#include "utils.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    rand_seed((unsigned int)time(NULL));

    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    // parse optional args: --data-dir <path> to load real images (PGM) or enable stb via CMake
    const char *data_dir = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
    }

    // prepare dataset
    const int N = 200;
    Image *images[N];
    int labels[N];
    if (!data_dir) {
        for (int i = 0; i < N; ++i) {
            int lab = i < N/2 ? 1 : 0; // first half cats
            images[i] = generate_synthetic(W, H, lab);
            labels[i] = lab;
        }
    } else {
        // Attempt to load files from directory (or a single file). We'll load up to N images.
        struct stat st;
        if (stat(data_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
            DIR *d = opendir(data_dir);
            if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
            struct dirent *ent; int idx = 0;
            while ((ent = readdir(d)) && idx < N) {
                if (ent->d_name[0] == '.') continue;
                // naive accept of image files
                char path[4096]; snprintf(path, sizeof(path), "%s/%s", data_dir, ent->d_name);
                Image *im = image_load_file(path, W, H);
                if (!im) continue;
                images[idx] = im;
                // for prototype: label by filename containing "cat" or "dog"
                if (strcasestr(ent->d_name, "cat")) labels[idx] = 1; else labels[idx] = 0;
                idx++;
            }
            closedir(d);
            if (idx == 0) { fprintf(stderr, "no images loaded from %s\n", data_dir); return 1; }
            // fill remainder with synthetic
            for (int i = idx; i < N; ++i) { int lab = i < N/2 ? 1 : 0; images[i] = generate_synthetic(W,H,lab); labels[i] = lab; }
        } else {
            // single file
            Image *im = image_load_file(data_dir, W, H);
            if (!im) { fprintf(stderr, "failed to load image %s\n", data_dir); return 1; }
            images[0] = im; labels[0] = 0; // default label
            // fill rest with synthetic
            for (int i = 1; i < N; ++i) { int lab = i < N/2 ? 1 : 0; images[i] = generate_synthetic(W,H,lab); labels[i] = lab; }
        }
    }

    const int epochs = 10;
    const float lr = 0.01f;
    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            float out = cnn_forward(net, images[i]);
            int pred = out > 0.5f;
            if (pred == labels[i]) correct++;
            float loss = cnn_backward_and_update(net, images[i], labels[i], lr);
            epoch_loss += loss;
        }
        printf("Epoch %d: loss=%.4f acc=%.3f\n", e+1, epoch_loss / N, (float)correct / N);
    }

    // save model
    cnn_save(net, "model.bin");
    printf("Saved tiny model to model.bin\n");

    for (int i = 0; i < N; ++i) image_free(images[i]);
    cnn_free(net);
    return 0;
}
