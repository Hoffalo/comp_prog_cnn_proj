/* Trainer main (restored)
 * Loads images from a directory layout like:
 *   PetImages/Cat/*.jpg
 *   PetImages/Dog/*.jpg
 * If no data directory is provided and PetImages isn't found, a small synthetic
 * dataset is used for quick testing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <stdbool.h>
#include "image.h"
#include "cnn.h"
#include "utils.h"
#include <math.h>

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    ext++;
    if (!ext) return 0;
    char low[8]; int i=0; for (; i<7 && ext[i]; ++i) low[i] = tolower((unsigned char)ext[i]); low[i]=0;
    return (strcmp(low,"jpg")==0 || strcmp(low,"jpeg")==0 || strcmp(low,"png")==0 || strcmp(low,"bmp")==0 || strcmp(low,"gif")==0 || strcmp(low,"pgm")==0 || strcmp(low,"tif")==0 || strcmp(low,"tiff")==0 || strcmp(low,"ppm")==0);
}

static void append_path(char ***paths, int **labs, int *count, int *cap, const char *p, int lab) {
    if (*count + 1 > *cap) {
        int nc = *cap ? (*cap * 2) : 256;
        *paths = realloc(*paths, sizeof(char*) * nc);
        *labs = realloc(*labs, sizeof(int) * nc);
        if (!*paths || !*labs) { fprintf(stderr, "out of memory\n"); exit(1); }
        *cap = nc;
    }
    (*paths)[*count] = strdup(p);
    (*labs)[*count] = lab;
    (*count)++;
}

int main(int argc, char **argv) {
    rand_seed((unsigned int)time(NULL));

    // Check for test mode
    bool test_mode = false;
    const char *test_image = NULL;
    bool dump_model = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test") == 0 && i+1 < argc) { test_mode = true; test_image = argv[i+1]; break; }
        if (strcmp(argv[i], "--dump-model") == 0) { dump_model = true; break; }
    }
    if (dump_model) {
        TinyCNN *net = cnn_create(16,16,4,3,2);
        if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }
        if (cnn_load(net, "model.bin") != 0) { fprintf(stderr, "failed to load model.bin\n"); cnn_free(net); return 1; }
        cnn_print_summary(net, 12, 12);
        cnn_free(net);
        return 0;
    }
    if (test_mode && test_image) {
        // Load model weights from model.bin
    TinyCNN *net = cnn_create(16, 16, 4, 3, 2);
    if (!net) { fprintf(stderr, "Failed to create network for inference\n"); return 1; }
    if (cnn_load(net, "model.bin") != 0) { fprintf(stderr, "Failed to load model.bin (ensure it was saved with matching architecture)\n"); cnn_free(net); return 1; }
        // Load and preprocess image
        Image *im = image_load_file(test_image, 16, 16);
        if (!im) { fprintf(stderr, "Failed to load test image: %s\n", test_image); cnn_free(net); return 1; }
        float out = cnn_forward(net, im);
        printf("Prediction for %s: %s (%.3f)\n", test_image, out > 0.5f ? "Cat" : "Dog", out);
        image_free(im); cnn_free(net);
        return 0;
    }

    const int W = 16, H = 16;
    const int filters = 8, ksize = 3, pool = 2; // increase filters for more capacity

    const char *data_dir = NULL;
    int max_images = 25000; // train on all images in PetImages
    int epochs = 10;
    float lr = 0.0005f;
    bool debug = false;
    float cli_l2 = -1.0f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        else if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images < 0) max_images = 0; i++; }
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) { epochs = atoi(argv[i+1]); if (epochs < 1) epochs = 1; i++; }
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) { lr = atof(argv[i+1]); if (lr <= 0.0f) lr = 0.01f; i++; }
        else if (strcmp(argv[i], "--debug") == 0) { debug = true; }
        else if (strcmp(argv[i], "--l2") == 0 && i+1 < argc) { cli_l2 = atof(argv[i+1]); i++; }
    }

    if (!data_dir) {
        struct stat st; if (stat("PetImages", &st) == 0 && S_ISDIR(st.st_mode)) data_dir = "PetImages";
    }

    char **paths = NULL;
    int *plabels = NULL;
    int pcount = 0, pcap = 0;

    // Balanced sampling from Cat and Dog folders
    if (data_dir) {
        char cat_dir[PATH_MAX], dog_dir[PATH_MAX];
        snprintf(cat_dir, sizeof(cat_dir), "%s/Cat", data_dir);
        snprintf(dog_dir, sizeof(dog_dir), "%s/Dog", data_dir);
        struct stat stcat, stdog;
        int cat_exists = (stat(cat_dir, &stcat) == 0 && S_ISDIR(stcat.st_mode));
        int dog_exists = (stat(dog_dir, &stdog) == 0 && S_ISDIR(stdog.st_mode));
        int per_class = max_images > 0 ? max_images / 2 : 0;
        if (cat_exists && dog_exists) {
            // Collect up to per_class images from each folder
            DIR *dcat = opendir(cat_dir);
            DIR *ddog = opendir(dog_dir);
            if (!dcat || !ddog) { fprintf(stderr, "failed to open Cat or Dog folder\n"); return 1; }
            struct dirent *ent;
            int cat_count = 0, dog_count = 0;
            while ((ent = readdir(dcat)) && (per_class == 0 || cat_count < per_class)) {
                if (ent->d_name[0] == '.') continue;
                if (!has_image_ext(ent->d_name)) continue;
                char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", cat_dir, ent->d_name);
                append_path(&paths, &plabels, &pcount, &pcap, fpath, 1);
                cat_count++;
            }
            while ((ent = readdir(ddog)) && (per_class == 0 || dog_count < per_class)) {
                if (ent->d_name[0] == '.') continue;
                if (!has_image_ext(ent->d_name)) continue;
                char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", dog_dir, ent->d_name);
                append_path(&paths, &plabels, &pcount, &pcap, fpath, 0);
                dog_count++;
            }
            closedir(dcat); closedir(ddog);
        } else {
            // Fallback: scan all subfolders as before
            struct stat st;
            if (stat(data_dir, &st) != 0) { fprintf(stderr, "data-dir '%s' not found\n", data_dir); return 1; }
            if (S_ISDIR(st.st_mode)) {
                DIR *d = opendir(data_dir);
                if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                struct dirent *ent;
                int has_subdirs = 0;
                while ((ent = readdir(d))) {
                    if (ent->d_name[0] == '.') continue;
                    char tmp[PATH_MAX]; snprintf(tmp, sizeof(tmp), "%s/%s", data_dir, ent->d_name);
                    struct stat st2; if (stat(tmp, &st2) == 0 && S_ISDIR(st2.st_mode)) { has_subdirs = 1; break; }
                }
                rewinddir(d);
                if (has_subdirs) {
                    while ((ent = readdir(d)) && (max_images == 0 || pcount < max_images)) {
                        if (ent->d_name[0] == '.') continue;
                        char sub[PATH_MAX]; snprintf(sub, sizeof(sub), "%s/%s", data_dir, ent->d_name);
                        struct stat st2; if (stat(sub, &st2) != 0) continue; if (!S_ISDIR(st2.st_mode)) continue;
                        int lab = 0; if (strcasestr(ent->d_name, "cat")) lab = 1; else if (strcasestr(ent->d_name, "dog")) lab = 0; else continue;
                        DIR *sd = opendir(sub); if (!sd) continue; struct dirent *fe;
                        while ((fe = readdir(sd)) && (max_images == 0 || pcount < max_images)) { if (fe->d_name[0] == '.') continue; if (!has_image_ext(fe->d_name)) continue; char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", sub, fe->d_name); append_path(&paths, &plabels, &pcount, &pcap, fpath, lab); }
                        closedir(sd);
                    }
                } else {
                    while ((ent = readdir(d)) && (max_images == 0 || pcount < max_images)) { if (ent->d_name[0] == '.') continue; if (!has_image_ext(ent->d_name)) continue; char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", data_dir, ent->d_name); int lab = strcasestr(ent->d_name, "cat") ? 1 : 0; append_path(&paths, &plabels, &pcount, &pcap, fpath, lab); }
                }
                closedir(d);
            } else {
                if (has_image_ext(data_dir)) append_path(&paths, &plabels, &pcount, &pcap, data_dir, 0);
            }
        }
    }

    // Validation split: 80% train, 20% val
    Image **images = malloc(sizeof(Image*) * (pcount ? pcount : 200));
    int *labels = malloc(sizeof(int) * (pcount ? pcount : 200));
    int N = 0;
    int N_val = 0;
    Image **val_images = NULL;
    int *val_labels = NULL;

    if (pcount == 0) {
        // synthetic fallback
        N = 200;
        for (int i = 0; i < N; ++i) { int lab = (i < N/2) ? 1 : 0; images[i] = generate_synthetic(W, H, lab); labels[i] = lab; }
        printf("Using synthetic dataset (%d samples)\n", N);
    } else {
        for (int i = 0; i < pcount; ++i) {
            Image *im = image_load_file(paths[i], W, H);
            free(paths[i]);
            if (!im) continue;
            images[N] = im; labels[N] = plabels[i]; N++;
        }
        free(paths); free(plabels);
        if (N == 0) { fprintf(stderr, "no valid images loaded from %s\n", data_dir); return 1; }
        printf("Loaded %d images from %s\n", N, data_dir ? data_dir : "(none)");
        int cats = 0, dogs = 0;
        for (int i = 0; i < N; ++i) { if (labels[i] == 1) ++cats; else ++dogs; }
        printf("Class distribution: cats=%d dogs=%d\n", cats, dogs);
        if (cats != dogs) {
            printf("WARNING: Class counts are not balanced!\n");
        }
        // Shuffle dataset to avoid validation split bias (we previously appended
        // all cats then all dogs which made the validation set contain only
        // one class). Do an in-place Fisher-Yates shuffle of images+labels.
        for (int i = N - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            Image *tmpi = images[i]; images[i] = images[j]; images[j] = tmpi;
            int tmpl = labels[i]; labels[i] = labels[j]; labels[j] = tmpl;
        }
        // Print first 10 loaded filenames and their labels
        printf("Sample loaded images and labels:\n");
        for (int i = 0; i < N && i < 10; ++i) {
            printf("  [%d] label=%s\n", i, labels[i] == 1 ? "cat" : "dog");
        }
    }

    // If --eval is passed, load model and evaluate on the loaded dataset (requires labels)
    bool eval_mode = false;
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--eval") == 0) { eval_mode = true; break; }
    bool gradcheck_mode = false;
    float grad_eps = 1e-4f;
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--gradcheck") == 0) { gradcheck_mode = true; }
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--grad-eps") == 0 && i+1 < argc) { grad_eps = atof(argv[i+1]); }
    if (eval_mode) {
        TinyCNN *net = cnn_create(W,H,filters,ksize,pool);
        if (!net) { fprintf(stderr, "failed to create network for eval\n"); return 1; }
        if (cnn_load(net, "model.bin") != 0) { fprintf(stderr, "failed to load model.bin for eval\n"); cnn_free(net); return 1; }
        int correct = 0; float totloss = 0.0f;
        int pred0 = 0, pred1 = 0; // counts of predicted Dog(0) and Cat(1)
        int TP = 0, TN = 0, FP = 0, FN = 0;
        // capture a few sample outputs for quick inspection
        const int SHOW = 10; float sample_out[SHOW]; int sample_label[SHOW]; int sample_pred[SHOW]; int shown = 0;
        for (int i = 0; i < N; ++i) {
            float out = cnn_forward(net, images[i]);
            int pred = out > 0.5f;
            if (pred == labels[i]) ++correct;
            if (pred) { pred1++; } else { pred0++; }
            if (labels[i] == 1 && pred == 1) TP++;
            if (labels[i] == 0 && pred == 0) TN++;
            if (labels[i] == 0 && pred == 1) FP++;
            if (labels[i] == 1 && pred == 0) FN++;
            // compute BCE loss for info
            if (labels[i]) totloss += -logf(out + 1e-8f); else totloss += -logf(1 - out + 1e-8f);
            if (shown < SHOW) { sample_out[shown] = out; sample_label[shown] = labels[i]; sample_pred[shown] = pred; shown++; }
        }
        printf("Eval on loaded data: accuracy=%.4f loss=%.6f (%d samples)\n", (float)correct / N, totloss / N, N);
        printf("Predictions: Cat=%d Dog=%d\n", pred1, pred0);
        printf("Confusion: TP=%d FP=%d FN=%d TN=%d\n", TP, FP, FN, TN);
        printf("Sample outputs (index 0..%d):\n", shown-1);
        for (int i = 0; i < shown; ++i) printf("  sample %d: out=%.6f label=%d pred=%d\n", i, sample_out[i], sample_label[i], sample_pred[i]);
        cnn_free(net);
        // cleanup images
        for (int i = 0; i < N; ++i) image_free(images[i]); free(images); free(labels);
        return 0;
    }

    if (gradcheck_mode) {
        if (N == 0) { fprintf(stderr, "No images loaded for gradcheck\n"); return 1; }
        TinyCNN *netc = cnn_create(W,H,filters,ksize,pool);
        if (!netc) { fprintf(stderr, "failed to create network for gradcheck\n"); return 1; }
        if (cnn_load(netc, "model.bin") != 0) fprintf(stderr, "note: model.bin not found or failed to load; using random init\n");
        printf("Running gradcheck on first image (index 0) with eps=%.6g\n", grad_eps);
        cnn_gradcheck(netc, images[0], labels[0], grad_eps);
        cnn_free(netc);
        for (int i = 0; i < N; ++i) image_free(images[i]); free(images); free(labels);
        return 0;
    }

    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    /* set debug flag in cnn implementation */
    extern int cnn_debug;
    if (debug) cnn_debug = 1;
    extern float cnn_l2;
    if (cli_l2 >= 0.0f) cnn_l2 = cli_l2;

    // Seed stdlib RNG for shuffling
    srand((unsigned)time(NULL));

    // training loop with per-epoch shuffling
    int *idx = malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    int best_epoch = 0;
    float best_val_acc = 0.0f;
    int patience = 10, wait = 0;
    // Split validation set (last 20%)
    N_val = N / 5;
    if (N_val > 0) {
        val_images = malloc(sizeof(Image*) * N_val);
        val_labels = malloc(sizeof(int) * N_val);
        for (int i = 0; i < N_val; ++i) {
            val_images[i] = images[N - N_val + i];
            val_labels[i] = labels[N - N_val + i];
        }
        N -= N_val;
    }
    for (int e = 0; e < epochs; ++e) {
        // Fisher-Yates shuffle
        for (int i = N-1; i > 0; --i) {
            int j = rand() % (i + 1);
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        float epoch_loss = 0.0f; int correct = 0;
        for (int it = 0; it < N; ++it) {
            int i = idx[it];
            float out = cnn_forward(net, images[i]); int pred = out > 0.5f;
            if (pred == labels[i]) correct++;
            float loss = cnn_backward_and_update(net, images[i], labels[i], lr);
            epoch_loss += loss;
        }
        // Validation accuracy
        int val_correct = 0;
        for (int i = 0; i < N_val; ++i) {
            float out = cnn_forward(net, val_images[i]);
            int pred = out > 0.5f;
            if (pred == val_labels[i]) val_correct++;
        }
    float val_acc = N_val > 0 ? (float)val_correct / N_val : 0.0f;
    printf("Epoch %d: loss=%.4f acc=%.3f val_acc=%.3f\n", e+1, epoch_loss / N, (float)correct / N, val_acc);
    // Print lightweight diagnostics for the final dense layer to detect bias drift
    float d2b=0.0f, d2mw=0.0f;
    cnn_get_dense2_stats(net, &d2b, &d2mw);
    printf("  dense2 stats: bias=%.6f mean_w=%.6f\n", d2b, d2mw);
        if (val_acc > best_val_acc) { best_val_acc = val_acc; best_epoch = e; wait = 0; }
        else wait++;
        if (wait >= patience) { printf("Early stopping at epoch %d (best val_acc=%.3f at epoch %d)\n", e+1, best_val_acc, best_epoch+1); break; }
    }
    free(idx);

    cnn_save(net, "model.bin"); printf("Saved tiny model to model.bin\n");
    for (int i = 0; i < N; ++i) image_free(images[i]);
    if (val_images) { for (int i = 0; i < N_val; ++i) image_free(val_images[i]); free(val_images); free(val_labels); }
    free(images); free(labels); cnn_free(net);
    return 0;
}
