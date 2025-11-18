/* Tiny trainer main
 * Loads images from PetImages/Cat and PetImages/Dog by default (labels derived from folder names).
 * Usage:
 *   ./bin/train                # uses PetImages if present, otherwise synthetic
 *   ./bin/train --data-dir DIR --max-images 1000
 */

/* Tiny trainer main
 * Loads images from PetImages/Cat and PetImages/Dog by default (labels derived from folder names).
 * Usage:
 *   ./bin/train                # uses PetImages if present, otherwise synthetic
 *   ./bin/train --data-dir DIR --max-images 1000
 */

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

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0; ext++;
    if (!ext) return 0;
    if (strcasecmp(ext, "jpg") == 0) return 1;
    if (strcasecmp(ext, "jpeg") == 0) return 1;
    if (strcasecmp(ext, "png") == 0) return 1;
    if (strcasecmp(ext, "bmp") == 0) return 1;
    if (strcasecmp(ext, "gif") == 0) return 1;
    if (strcasecmp(ext, "pgm") == 0) return 1;
    if (strcasecmp(ext, "tif") == 0) return 1;
    if (strcasecmp(ext, "tiff") == 0) return 1;
    if (strcasecmp(ext, "ppm") == 0) return 1;
    return 0;
}

static void ensure_capacity(Image ***imgs, int **labs, int *cap, int need) {
    if (need <= *cap) return;
    int newcap = *cap ? *cap * 2 : 256;
    while (newcap < need) newcap *= 2;
    Image **nimgs = realloc(*imgs, sizeof(Image*) * newcap);
    int *nlabs = realloc(*labs, sizeof(int) * newcap);
    if (!nimgs || !nlabs) { fprintf(stderr, "out of memory\n"); exit(1); }
    *imgs = nimgs; *labs = nlabs; *cap = newcap;
}

int main(int argc, char **argv) {
    rand_seed((unsigned int)time(NULL));

    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;

    const char *data_dir = NULL;
    int max_images = 1000; // default cap
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        else if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images <= 0) max_images = 1000; i++; }
    }

    // prefer PetImages in repo root if present
    struct stat stroot;
    if (!data_dir) {
        if (stat("PetImages", &stroot) == 0 && S_ISDIR(stroot.st_mode)) data_dir = "PetImages";
    }

    Image **images = NULL;
    int *labels = NULL;
    int count = 0, cap = 0;

    if (!data_dir) {
        // fallback synthetic small dataset
        int N = 200;
        ensure_capacity(&images, &labels, &cap, N);
        for (int i = 0; i < N; ++i) {
            images[count] = generate_synthetic(W, H, i < N/2);
            labels[count] = (i < N/2) ? 1 : 0;
            count++;
        }
        printf("Using synthetic dataset (%d samples)\n", count);
    } else {
        struct stat st;
        if (stat(data_dir, &st) != 0) { fprintf(stderr, "data-dir '%s' not found\n", data_dir); return 1; }
        if (S_ISDIR(st.st_mode)) {
            // check whether data_dir contains subdirectories (class folders)
            DIR *droot = opendir(data_dir);
            if (!droot) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
            struct dirent *entroot;
            int found_subdirs = 0;
            while ((entroot = readdir(droot))) {
                if (entroot->d_name[0] == '.') continue;
                char path[4096]; snprintf(path, sizeof(path), "%s/%s", data_dir, entroot->d_name);
                struct stat st2; if (stat(path, &st2) != 0) continue;
                if (S_ISDIR(st2.st_mode)) { found_subdirs = 1; break; }
            }
            closedir(droot);

            if (found_subdirs) {
                DIR *d = opendir(data_dir);
                if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                while ((entroot = readdir(d))) {
                    if (entroot->d_name[0] == '.') continue;
                    char subpath[4096]; snprintf(subpath, sizeof(subpath), "%s/%s", data_dir, entroot->d_name);
                    struct stat st3; if (stat(subpath, &st3) != 0) continue;
                    if (!S_ISDIR(st3.st_mode)) continue;
                    int lab = -1;
                    if (strcasestr(entroot->d_name, "cat")) lab = 1;
                    else if (strcasestr(entroot->d_name, "dog")) lab = 0;
                    else continue; // skip unknown folders

                    DIR *sd = opendir(subpath);
                    if (!sd) continue;
                    struct dirent *fent;
                    while ((fent = readdir(sd))) {
                        if (fent->d_name[0] == '.') continue;
                        if (!has_image_ext(fent->d_name)) continue;
                        if (max_images > 0 && count >= max_images) break;
                        char filepath[4096]; snprintf(filepath, sizeof(filepath), "%s/%s", subpath, fent->d_name);
                        Image *im = image_load_file(filepath, W, H);
                        if (!im) continue;
                        ensure_capacity(&images, &labels, &cap, count+1);
                        images[count] = im; labels[count] = lab; count++;
                    }
                    closedir(sd);
                    if (max_images > 0 && count >= max_images) break;
                }
                closedir(d);
            } else {
                // load files directly in data_dir and label by filename containing 'cat'
                DIR *d = opendir(data_dir);
                if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                struct dirent *ent2;
                while ((ent2 = readdir(d))) {
                    if (ent2->d_name[0] == '.') continue;
                    if (!has_image_ext(ent2->d_name)) continue;
                    if (max_images > 0 && count >= max_images) break;
                    char path2[4096]; snprintf(path2, sizeof(path2), "%s/%s", data_dir, ent2->d_name);
                    Image *im = image_load_file(path2, W, H);
                    if (!im) continue;
                    int lab = strcasestr(ent2->d_name, "cat") ? 1 : 0;
                    ensure_capacity(&images, &labels, &cap, count+1);
                    images[count] = im; labels[count] = lab; count++;
                }
                closedir(d);
            }
        } else {
            // single file
            Image *im = image_load_file(data_dir, W, H);
            if (!im) { fprintf(stderr, "failed to load image %s\n", data_dir); return 1; }
            ensure_capacity(&images, &labels, &cap, count+1);
            images[count] = im; labels[count] = 0; count++;
        }

        if (count == 0) { fprintf(stderr, "no images loaded from %s\n", data_dir); return 1; }
        printf("Loaded %d images from %s\n", count, data_dir);
    }

    // create network (now that we have data)
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    // training loop
    const int epochs = 10;
    const float lr = 0.01f;
    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < count; ++i) {
            float out = cnn_forward(net, images[i]);
            int pred = out > 0.5f;
            if (pred == labels[i]) correct++;
            float loss = cnn_backward_and_update(net, images[i], labels[i], lr);
            epoch_loss += loss;
        }
        printf("Epoch %d: loss=%.4f acc=%.3f\n", e+1, epoch_loss / count, (float)correct / count);
    }

    // save model and cleanup
    cnn_save(net, "model.bin");
    printf("Saved tiny model to model.bin\n");
    for (int i = 0; i < count; ++i) image_free(images[i]);
    free(images); free(labels);
    cnn_free(net);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include "image.h"
#include "cnn.h"
#include "utils.h"

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    ext++;
    if (!ext) return 0;
    if (strcasecmp(ext, "jpg") == 0) return 1;
    if (strcasecmp(ext, "jpeg") == 0) return 1;
    if (strcasecmp(ext, "png") == 0) return 1;
    if (strcasecmp(ext, "bmp") == 0) return 1;
    if (strcasecmp(ext, "gif") == 0) return 1;
    if (strcasecmp(ext, "pgm") == 0) return 1;
    if (strcasecmp(ext, "tif") == 0) return 1;
    if (strcasecmp(ext, "tiff") == 0) return 1;
    if (strcasecmp(ext, "ppm") == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    rand_seed((unsigned int)time(NULL));

    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;

    // parse args
    const char *data_dir = NULL;
    int max_images = 1000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        else if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images <= 0) max_images = 1000; i++; }
    }

    // if no explicit data-dir, check for PetImages in repo root
    if (!data_dir) {
        struct stat st;
        if (stat("PetImages", &st) == 0 && S_ISDIR(st.st_mode)) data_dir = "PetImages";
    }

    // Build list of image file paths and labels (0=dog,1=cat)
    char **paths = NULL;
    int *plabels = NULL;
    int pcount = 0;

    if (data_dir) {
        struct stat st;
        if (stat(data_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
            DIR *d = opendir(data_dir);
            if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
            struct dirent *ent;
            while ((ent = readdir(d)) && pcount < max_images) {
                if (ent->d_name[0] == '.') continue;
                // check if subdir
                char subpath[PATH_MAX]; snprintf(subpath, sizeof(subpath), "%s/%s", data_dir, ent->d_name);
                struct stat st2;
                if (stat(subpath, &st2) != 0) continue;
                if (S_ISDIR(st2.st_mode)) {
                    // decide label by folder name
                    int label = 0;
                    if (strcasestr(ent->d_name, "cat")) label = 1;
                    else if (strcasestr(ent->d_name, "dog")) label = 0;
                    else label = 0;
                    DIR *sd = opendir(subpath);
                    if (!sd) continue;
                    struct dirent *fe;
                    while ((fe = readdir(sd)) && pcount < max_images) {
                        if (fe->d_name[0] == '.') continue;
                        if (!has_image_ext(fe->d_name)) continue;
                        char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", subpath, fe->d_name);
                        paths = realloc(paths, sizeof(char*) * (pcount+1));
                        plabels = realloc(plabels, sizeof(int) * (pcount+1));
                        paths[pcount] = strdup(fpath);
                        plabels[pcount] = label;
                        pcount++;
                    }
                    closedir(sd);
                } else {
                    // file directly in data_dir
                    if (!has_image_ext(ent->d_name)) continue;
                    char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", data_dir, ent->d_name);
                    int label = strcasestr(ent->d_name, "cat") ? 1 : 0;
                    paths = realloc(paths, sizeof(char*) * (pcount+1));
                    plabels = realloc(plabels, sizeof(int) * (pcount+1));
                    paths[pcount] = strdup(fpath);
                    plabels[pcount] = label;
                    pcount++;
                }
            }
            closedir(d);
        } else {
            // data_dir is a single file
            if (has_image_ext(data_dir)) {
                paths = malloc(sizeof(char*)); plabels = malloc(sizeof(int));
                paths[0] = strdup(data_dir); plabels[0] = 0; pcount = 1;
            }
        }
    }

    Image **images = NULL;
    int *labels = NULL;
    int N = 0;

    if (pcount == 0) {
        // fallback to synthetic
        N = 200;
        images = malloc(sizeof(Image*) * N);
        labels = malloc(sizeof(int) * N);
        for (int i = 0; i < N; ++i) { int lab = i < N/2 ? 1 : 0; images[i] = generate_synthetic(W,H,lab); labels[i] = lab; }
    } else {
        images = malloc(sizeof(Image*) * pcount);
        labels = malloc(sizeof(int) * pcount);
        for (int i = 0; i < pcount; ++i) {
            Image *im = image_load_file(paths[i], W, H);
            free(paths[i]);
            if (!im) continue;
            images[N] = im; labels[N] = plabels[i]; N++;
        }
        free(paths); free(plabels);
        if (N == 0) {
            fprintf(stderr, "no valid images loaded from %s\n", data_dir);
            return 1;
        }
    }

    // create network after we have N (we could create earlier too)
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

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

    cnn_save(net, "model.bin");
    printf("Saved tiny model to model.bin\n");

    for (int i = 0; i < N; ++i) image_free(images[i]);
    free(images); free(labels);
    cnn_free(net);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <ctype.h>
#include "image.h"
#include "cnn.h"
#include "utils.h"

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0; ext++;
    if (!ext) return 0;
    if (strcasecmp(ext, "jpg") == 0) return 1;
    if (strcasecmp(ext, "jpeg") == 0) return 1;
    if (strcasecmp(ext, "png") == 0) return 1;
    if (strcasecmp(ext, "bmp") == 0) return 1;
    if (strcasecmp(ext, "gif") == 0) return 1;
    if (strcasecmp(ext, "pgm") == 0) return 1;
    if (strcasecmp(ext, "tif") == 0) return 1;
    if (strcasecmp(ext, "tiff") == 0) return 1;
    if (strcasecmp(ext, "ppm") == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    rand_seed((unsigned int)time(NULL));

    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;

    // parse args
    const char *data_dir = NULL;
    int max_images = 0; // 0 = unlimited
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        else if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images <= 0) max_images = 0; i++; }
    }

    // if no explicit data-dir, check for PetImages in repo root
    if (!data_dir) {
        struct stat st; if (stat("PetImages", &st) == 0 && S_ISDIR(st.st_mode)) data_dir = "PetImages";
    }

    // Build list of file paths and labels
    char **paths = NULL; int *plabels = NULL; int pcount = 0; int pcap = 0;
    const int HARD_CAP = 20000;

    if (data_dir) {
        struct stat st; if (stat(data_dir, &st) != 0) { fprintf(stderr, "data-dir '%s' not found\n", data_dir); return 1; }
        if (S_ISDIR(st.st_mode)) {
            DIR *d = opendir(data_dir); if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
            struct dirent *ent;
            while ((ent = readdir(d))) {
                if (ent->d_name[0] == '.') continue;
                char subpath[4096]; snprintf(subpath, sizeof(subpath), "%s/%s", data_dir, ent->d_name);
                struct stat st2; if (stat(subpath, &st2) != 0) continue;
                if (S_ISDIR(st2.st_mode)) {
                    // subdir -> iterate files
                    int lab = 0; if (strcasestr(ent->d_name, "cat")) lab = 1; else if (strcasestr(ent->d_name, "dog")) lab = 0; else lab = 0;
                    DIR *sd = opendir(subpath); if (!sd) continue;
                    struct dirent *fe;
                    while ((fe = readdir(sd))) {
                        if (fe->d_name[0] == '.') continue;
                        if (!has_image_ext(fe->d_name)) continue;
                        if (max_images > 0 && pcount >= max_images) break;
                        if (pcount >= HARD_CAP) break;
                        if (pcount + 1 > pcap) { pcap = pcap ? pcap*2 : 1024; paths = realloc(paths, sizeof(char*)*pcap); plabels = realloc(plabels, sizeof(int)*pcap); }
                        char *fp = malloc(4096);
                        snprintf(fp, 4096, "%s/%s", subpath, fe->d_name);
                        paths[pcount] = fp; plabels[pcount] = lab; pcount++;
                    }
                    closedir(sd);
                } else {
                    // file directly in data_dir
                    if (!has_image_ext(ent->d_name)) continue;
                    if (max_images > 0 && pcount >= max_images) break;
                    if (pcount >= HARD_CAP) break;
                    if (pcount + 1 > pcap) { pcap = pcap ? pcap*2 : 1024; paths = realloc(paths, sizeof(char*)*pcap); plabels = realloc(plabels, sizeof(int)*pcap); }
                    char *fp = malloc(4096);
                    snprintf(fp, 4096, "%s/%s", data_dir, ent->d_name);
                    int lab = strcasestr(ent->d_name, "cat") ? 1 : 0;
                    paths[pcount] = fp; plabels[pcount] = lab; pcount++;
                }
                if (max_images > 0 && pcount >= max_images) break;
                if (pcount >= HARD_CAP) break;
            }
            closedir(d);
        } else {
            // single file
            if (has_image_ext(data_dir)) {
                paths = malloc(sizeof(char*)); plabels = malloc(sizeof(int)); paths[0] = strdup(data_dir); plabels[0] = 0; pcount = 1;
            }
        }
    }

    Image **images = NULL; int *labels = NULL; int N = 0;
    if (pcount == 0) {
        // fallback synthetic
        N = 200; images = malloc(sizeof(Image*)*N); labels = malloc(sizeof(int)*N);
        for (int i = 0; i < N; ++i) { images[i] = generate_synthetic(W,H,i < N/2); labels[i] = (i < N/2) ? 1 : 0; }
        printf("Using synthetic dataset (%d samples)\n", N);
    } else {
        images = malloc(sizeof(Image*) * pcount); labels = malloc(sizeof(int) * pcount);
        for (int i = 0; i < pcount; ++i) {
            Image *im = image_load_file(paths[i], W, H);
            free(paths[i]);
            if (!im) continue;
            images[N] = im; labels[N] = plabels[i]; N++;
        }
        free(paths); free(plabels);
        if (N == 0) { fprintf(stderr, "no valid images loaded from %s\n", data_dir); return 1; }
        printf("Loaded %d images from %s\n", N, data_dir);
    }

    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    const int epochs = 10; const float lr = 0.01f;
    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f; int correct = 0;
        for (int i = 0; i < N; ++i) {
            float out = cnn_forward(net, images[i]); int pred = out > 0.5f;
            if (pred == labels[i]) correct++;
            float loss = cnn_backward_and_update(net, images[i], labels[i], lr);
            epoch_loss += loss;
        }
        printf("Epoch %d: loss=%.4f acc=%.3f\n", e+1, epoch_loss / N, (float)correct / N);
    }

    cnn_save(net, "model.bin"); printf("Saved tiny model to model.bin\n");
    for (int i = 0; i < N; ++i) image_free(images[i]); free(images); free(labels); cnn_free(net);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <ctype.h>
#include "image.h"
#include "cnn.h"
#include "utils.h"

// Simple dynamic array helpers
static void ensure_capacity(Image ***imgs, int **labs, int *cap, int need) {
    if (need <= *cap) return;
    int newcap = *cap ? *cap * 2 : 256;
    while (newcap < need) newcap *= 2;
    Image **nimgs = realloc(*imgs, sizeof(Image*) * newcap);
    int *nlabs = realloc(*labs, sizeof(int) * newcap);
    if (!nimgs || !nlabs) {
        fprintf(stderr, "out of memory\n"); exit(1);
    }
    *imgs = nimgs; *labs = nlabs; *cap = newcap;
}

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    ext++;
    char low[8]; int i=0; for (; i<7 && ext[i]; ++i) low[i] = tolower((unsigned char)ext[i]); low[i]=0;
    return (strcmp(low,"jpg")==0 || strcmp(low,"jpeg")==0 || strcmp(low,"png")==0 || strcmp(low,"bmp")==0 || strcmp(low,"tga")==0 || strcmp(low,"gif")==0 || strcmp(low,"pgm")==0 || strcmp(low,"tif")==0 || strcmp(low,"tiff")==0);
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0; ext++;
    if (!ext) return 0;
    if (strcasecmp(ext, "jpg") == 0) return 1;
    if (strcasecmp(ext, "jpeg") == 0) return 1;
    if (strcasecmp(ext, "png") == 0) return 1;
    if (strcasecmp(ext, "bmp") == 0) return 1;
    if (strcasecmp(ext, "gif") == 0) return 1;
    if (strcasecmp(ext, "pgm") == 0) return 1;
    if (strcasecmp(ext, "tif") == 0) return 1;
    if (strcasecmp(ext, "tiff") == 0) return 1;
    if (strcasecmp(ext, "ppm") == 0) return 1;
    return 0;
}

static void ensure_capacity(Image ***imgs, int **labs, int *cap, int need) {
    if (need <= *cap) return;
    int newcap = *cap ? *cap * 2 : 256;
    while (newcap < need) newcap *= 2;
    Image **nimgs = realloc(*imgs, sizeof(Image*) * newcap);
    int *nlabs = realloc(*labs, sizeof(int) * newcap);
    if (!nimgs || !nlabs) { fprintf(stderr, "out of memory\n"); exit(1); }
    *imgs = nimgs; *labs = nlabs; *cap = newcap;
}

int main(int argc, char **argv) {
    rand_seed((unsigned int)time(NULL));

    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;

    const char *data_dir = NULL;
    int max_images = 1000; // default cap
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        else if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images <= 0) max_images = 1000; i++; }
    }

    // prefer PetImages in repo root if present
    struct stat stroot;
    if (!data_dir) {
        if (stat("PetImages", &stroot) == 0 && S_ISDIR(stroot.st_mode)) data_dir = "PetImages";
    }

    Image **images = NULL;
    int *labels = NULL;
    int count = 0, cap = 0;

    if (!data_dir) {
        // fallback synthetic small dataset
        int N = 200;
        ensure_capacity(&images, &labels, &cap, N);
        for (int i = 0; i < N; ++i) {
            images[count] = generate_synthetic(W, H, i < N/2);
            labels[count] = (i < N/2) ? 1 : 0;
            count++;
        }
        printf("Using synthetic dataset (%d samples)\n", count);
    } else {
        struct stat st;
        if (stat(data_dir, &st) != 0) { fprintf(stderr, "data-dir '%s' not found\n", data_dir); return 1; }
        if (S_ISDIR(st.st_mode)) {
            // check whether data_dir contains subdirectories (class folders)
            DIR *droot = opendir(data_dir);
            if (!droot) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
            struct dirent *entroot;
            int found_subdirs = 0;
            while ((entroot = readdir(droot))) {
                if (entroot->d_name[0] == '.') continue;
                char path[4096]; snprintf(path, sizeof(path), "%s/%s", data_dir, entroot->d_name);
                struct stat st2; if (stat(path, &st2) != 0) continue;
                if (S_ISDIR(st2.st_mode)) { found_subdirs = 1; break; }
            }
            closedir(droot);

            if (found_subdirs) {
                DIR *d = opendir(data_dir);
                if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                while ((entroot = readdir(d))) {
                    if (entroot->d_name[0] == '.') continue;
                    char subpath[4096]; snprintf(subpath, sizeof(subpath), "%s/%s", data_dir, entroot->d_name);
                    struct stat st3; if (stat(subpath, &st3) != 0) continue;
                    if (!S_ISDIR(st3.st_mode)) continue;
                    int lab = -1;
                    if (strcasestr(entroot->d_name, "cat")) lab = 1;
                    else if (strcasestr(entroot->d_name, "dog")) lab = 0;
                    else continue; // skip unknown folders

                    DIR *sd = opendir(subpath);
                    if (!sd) continue;
                    struct dirent *fent;
                    while ((fent = readdir(sd))) {
                        if (fent->d_name[0] == '.') continue;
                        if (!has_image_ext(fent->d_name)) continue;
                        if (max_images > 0 && count >= max_images) break;
                        char filepath[4096]; snprintf(filepath, sizeof(filepath), "%s/%s", subpath, fent->d_name);
                        Image *im = image_load_file(filepath, W, H);
                        if (!im) continue;
                        ensure_capacity(&images, &labels, &cap, count+1);
                        images[count] = im; labels[count] = lab; count++;
                    }
                    closedir(sd);
                    if (max_images > 0 && count >= max_images) break;
                }
                closedir(d);
            } else {
                // load files directly in data_dir and label by filename containing 'cat'
                DIR *d = opendir(data_dir);
                if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                struct dirent *ent2;
                while ((ent2 = readdir(d))) {
                    if (ent2->d_name[0] == '.') continue;
                    if (!has_image_ext(ent2->d_name)) continue;
                    if (max_images > 0 && count >= max_images) break;
                    char path2[4096]; snprintf(path2, sizeof(path2), "%s/%s", data_dir, ent2->d_name);
                    Image *im = image_load_file(path2, W, H);
                    if (!im) continue;
                    int lab = strcasestr(ent2->d_name, "cat") ? 1 : 0;
                    ensure_capacity(&images, &labels, &cap, count+1);
                    images[count] = im; labels[count] = lab; count++;
                }
                closedir(d);
            }
        } else {
            // single file
            Image *im = image_load_file(data_dir, W, H);
            if (!im) { fprintf(stderr, "failed to load image %s\n", data_dir); return 1; }
            ensure_capacity(&images, &labels, &cap, count+1);
            images[count] = im; labels[count] = 0; count++;
        }

        if (count == 0) { fprintf(stderr, "no images loaded from %s\n", data_dir); return 1; }
        printf("Loaded %d images from %s\n", count, data_dir);
    }

    // create network (now that we have data)
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    // training loop
    const int epochs = 10;
    const float lr = 0.01f;
    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < count; ++i) {
            float out = cnn_forward(net, images[i]);
            int pred = out > 0.5f;
            if (pred == labels[i]) correct++;
            float loss = cnn_backward_and_update(net, images[i], labels[i], lr);
            epoch_loss += loss;
        }
        printf("Epoch %d: loss=%.4f acc=%.3f\n", e+1, epoch_loss / count, (float)correct / count);
    }

    // save model and cleanup
    cnn_save(net, "model.bin");
    printf("Saved tiny model to model.bin\n");
    for (int i = 0; i < count; ++i) image_free(images[i]);
    free(images); free(labels);
    cnn_free(net);
    return 0;
}

                } else {
                    // file directly in data_dir
                    if (!has_image_ext(ent->d_name)) continue;
                    char fpath[PATH_MAX]; snprintf(fpath, sizeof(fpath), "%s/%s", data_dir, ent->d_name);
                    int label = strcasestr(ent->d_name, "cat") ? 1 : 0;
                    paths = realloc(paths, sizeof(char*) * (pcount+1));
                    plabels = realloc(plabels, sizeof(int) * (pcount+1));
                    paths[pcount] = strdup(fpath);
                    plabels[pcount] = label;
                    pcount++;
                }
            }
            closedir(d);
        } else {
            // data_dir is a single file
            if (has_image_ext(data_dir)) {
                paths = malloc(sizeof(char*)); plabels = malloc(sizeof(int));
                paths[0] = strdup(data_dir); plabels[0] = 0; pcount = 1;
            }
        }
    }

    Image **images = NULL;
    int *labels = NULL;
    int N = 0;

    if (pcount == 0) {
        // fallback to synthetic
        N = 200;
        images = malloc(sizeof(Image*) * N);
        labels = malloc(sizeof(int) * N);
        for (int i = 0; i < N; ++i) { int lab = i < N/2 ? 1 : 0; images[i] = generate_synthetic(W,H,lab); labels[i] = lab; }
    } else {
        images = malloc(sizeof(Image*) * pcount);
        labels = malloc(sizeof(int) * pcount);
        for (int i = 0; i < pcount; ++i) {
            Image *im = image_load_file(paths[i], W, H);
            free(paths[i]);
            if (!im) continue;
            images[N] = im; labels[N] = plabels[i]; N++;
        }
        free(paths); free(plabels);
        if (N == 0) {
            fprintf(stderr, "no valid images loaded from %s\n", data_dir);
            return 1;
        }
    }

    // create network after we have N (we could create earlier too)
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

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

    cnn_save(net, "model.bin");
    printf("Saved tiny model to model.bin\n");

    for (int i = 0; i < N; ++i) image_free(images[i]);
    free(images); free(labels);
    cnn_free(net);
    return 0;
}
/* Tiny trainer main
 * - By default, if a `PetImages` directory exists in the repo root, the program
 *   will load images from its subdirectories (e.g. `PetImages/Cat`, `PetImages/Dog`)
 *   and label images by subfolder name (folders containing "cat" -> label=1, "dog" -> 0).
 * - You can override with `--data-dir <path>` and limit number with `--max-images N`.
 */

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

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0; ext++;
    if (!ext) return 0;
    if (strcasecmp(ext, "jpg") == 0) return 1;
    if (strcasecmp(ext, "jpeg") == 0) return 1;
    if (strcasecmp(ext, "png") == 0) return 1;
    if (strcasecmp(ext, "bmp") == 0) return 1;
    if (strcasecmp(ext, "gif") == 0) return 1;
    if (strcasecmp(ext, "pgm") == 0) return 1;
    if (strcasecmp(ext, "tif") == 0) return 1;
    if (strcasecmp(ext, "tiff") == 0) return 1;
    if (strcasecmp(ext, "ppm") == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    rand_seed((unsigned int)time(NULL));

    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    const char *data_dir = NULL;
    int max_images = 1000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        else if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images <= 0) max_images = 1000; i++; }
    }

    // prefer PetImages in repo root if present
    struct stat stroot;
    if (!data_dir) {
        if (stat("PetImages", &stroot) == 0 && S_ISDIR(stroot.st_mode)) data_dir = "PetImages";
    }

    int Ncap = max_images;
    Image **images = malloc(sizeof(Image*) * Ncap);
    int *labels = malloc(sizeof(int) * Ncap);
    if (!images || !labels) { fprintf(stderr, "out of memory\n"); return 1; }
    int loaded = 0;

    if (!data_dir) {
        // no data dir: synthetic
        for (int i = 0; i < Ncap; ++i) {
            int lab = i < Ncap/2 ? 1 : 0;
            images[i] = generate_synthetic(W, H, lab);
            labels[i] = lab;
            loaded++;
        }
    } else {
        // data_dir exists: if directory, scan subfolders and use folder name as label
        struct stat st;
        if (stat(data_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
            DIR *rootd = opendir(data_dir);
            if (!rootd) { fprintf(stderr, "failed to open data dir %s: %s\n", data_dir, strerror(errno)); return 1; }
            struct dirent *ent;
            while ((ent = readdir(rootd)) && loaded < Ncap) {
                if (ent->d_name[0] == '.') continue;
                char classpath[4096]; snprintf(classpath, sizeof(classpath), "%s/%s", data_dir, ent->d_name);
                struct stat cst;
                if (stat(classpath, &cst) != 0) continue;
                if (!S_ISDIR(cst.st_mode)) continue;
                int lab = -1;
                if (strcasestr(ent->d_name, "cat")) lab = 1;
                else if (strcasestr(ent->d_name, "dog")) lab = 0;
                else continue; // skip unknown folders

                DIR *cd = opendir(classpath);
                if (!cd) continue;
                struct dirent *fent;
                while ((fent = readdir(cd)) && loaded < Ncap) {
                    if (fent->d_name[0] == '.') continue;
                    if (!has_image_ext(fent->d_name)) continue;
                    char path[4096]; snprintf(path, sizeof(path), "%s/%s", classpath, fent->d_name);
                    Image *im = image_load_file(path, W, H);
                    if (!im) continue;
                    images[loaded] = im;
                    labels[loaded] = lab;
                    loaded++;
                }
                closedir(cd);
            }
            closedir(rootd);
        } else {
            // single file
            Image *im = image_load_file(data_dir, W, H);
            if (!im) { fprintf(stderr, "failed to load image %s\n", data_dir); return 1; }
            images[loaded] = im; labels[loaded] = 0; loaded++;
        }

        if (loaded == 0) {
            fprintf(stderr, "no images loaded from %s; falling back to synthetic\n", data_dir);
            for (int i = 0; i < Ncap; ++i) {
                int lab = i < Ncap/2 ? 1 : 0;
                images[i] = generate_synthetic(W, H, lab);
                labels[i] = lab;
                loaded++;
            }
        }
        // If fewer than Ncap, pad with synthetic to fill arrays
        for (int i = loaded; i < Ncap; ++i) {
            int lab = i < Ncap/2 ? 1 : 0;
            images[i] = generate_synthetic(W, H, lab);
            labels[i] = lab;
            loaded++;
        }
    }

    const int epochs = 10;
    const float lr = 0.01f;
    int N = loaded;
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

    cnn_save(net, "model.bin");
    printf("Saved tiny model to model.bin\n");

    for (int i = 0; i < N; ++i) image_free(images[i]);
    free(images); free(labels);
    cnn_free(net);
    return 0;
}
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

    // parse optional args: --data-dir <path> to load real images (expects subdirs per class)
    // and --max-images <n> to limit how many images to load (default 1000)
    const int W = 16, H = 16;
    const int filters = 4, ksize = 3, pool = 2;
    TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
    if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

    // parse optional args: --data-dir <path> to load real images OR use default `PetImages`
    // also support --max-images <n>
    const char *data_dir = NULL;
    int max_images = 1000; // default cap to avoid memory blowups
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
        if (strcmp(argv[i], "--max-images") == 0 && i+1 < argc) { max_images = atoi(argv[i+1]); if (max_images <= 0) max_images = 1000; i++; }
    }

    // if no explicit data-dir, prefer PetImages in project root if it exists
    struct stat stroot;
    if (!data_dir) {
        if (stat("PetImages", &stroot) == 0 && S_ISDIR(stroot.st_mode)) data_dir = "PetImages";
    }

    // prepare dataset (arrays sized to max_images)
    const int N = max_images;
    Image *images[N];
    int labels[N];
    for (int i = 0; i < N; ++i) { images[i] = NULL; labels[i] = 0; }

    // helper: check extension
    int has_image_ext(const char *name) {
        const char *ext = strrchr(name, '.');
        if (!ext) return 0;
        ext++;
        if (!ext) return 0;
        if (strcasecmp(ext, "jpg") == 0) return 1;
        if (strcasecmp(ext, "jpeg") == 0) return 1;
        if (strcasecmp(ext, "png") == 0) return 1;
        if (strcasecmp(ext, "bmp") == 0) return 1;
        if (strcasecmp(ext, "gif") == 0) return 1;
        if (strcasecmp(ext, "pgm") == 0) return 1;
        if (strcasecmp(ext, "tif") == 0) return 1;
        if (strcasecmp(ext, "tiff") == 0) return 1;
        if (strcasecmp(ext, "ppm") == 0) return 1;
        return 0;
    }

    int loaded = 0;
    if (!data_dir) {
        // fallback: synthetic data if no data_dir and no PetImages
        for (int i = 0; i < N; ++i) {
            int lab = i < N/2 ? 1 : 0; // first half cats
            images[i] = generate_synthetic(W, H, lab);
            rand_seed((unsigned int)time(NULL));

            const int W = 16, H = 16;
            const int filters = 4, ksize = 3, pool = 2;
            TinyCNN *net = cnn_create(W, H, filters, ksize, pool);
            if (!net) { fprintf(stderr, "failed to create network\n"); return 1; }

            // parse optional args: --data-dir <path> to load real images (PGM/stb)
            // optional: --max-samples <n> (0 = unlimited, capped internally)
            const char *data_dir = NULL;
            int max_samples = 0; // 0 = unlimited
            for (int i = 1; i < argc; ++i) {
                if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) { data_dir = argv[i+1]; i++; }
                else if (strcmp(argv[i], "--max-samples") == 0 && i+1 < argc) { max_samples = atoi(argv[i+1]); i++; }
            }

            // dynamic dataset containers
            Image **images = NULL;
            int *labels = NULL;
            int count = 0, cap = 0;
            void grow_fn(void) { if (count+1 > cap) { cap = cap ? cap*2 : 256; images = realloc(images, sizeof(Image*)*cap); labels = realloc(labels, sizeof(int)*cap); } }

            // helper: check extension
            int has_ext_fn(const char *name) {
                const char *ext = strrchr(name, '.');
                if (!ext) return 0;
                ++ext;
                char low[8]; int i=0; for (; i<7 && ext[i]; ++i) low[i] = tolower((unsigned char)ext[i]); low[i]=0;
                return (strcmp(low,"jpg")==0 || strcmp(low,"jpeg")==0 || strcmp(low,"png")==0 || strcmp(low,"bmp")==0 || strcmp(low,"tga")==0 || strcmp(low,"gif")==0 || strcmp(low,"pgm")==0);
            }

            // internal safety cap to avoid OOM when thousands of large images exist
            const int HARD_CAP = 20000;

            if (!data_dir) {
                // synthetic data fallback: keep original fixed N=200
                const int N = 200;
                for (int i = 0; i < N; ++i) {
                    grow_fn();
                    int lab = i < N/2 ? 1 : 0;
                    images[count] = generate_synthetic(W, H, lab);
                    labels[count] = lab;
                    count++;
                }
            } else {
                struct stat st;
                if (stat(data_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
                    // check for subdirectories (e.g., Cat and Dog)
                    DIR *droot = opendir(data_dir);
                    if (!droot) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                    struct dirent *entroot;
                    int found_subdirs = 0;
                    // First pass: detect subfolders
                    while ((entroot = readdir(droot))) {
                        if (entroot->d_name[0] == '.') continue;
                        char path[4096]; snprintf(path, sizeof(path), "%s/%s", data_dir, entroot->d_name);
                        struct stat st2; if (stat(path, &st2) != 0) continue;
                        if (S_ISDIR(st2.st_mode)) { found_subdirs = 1; break; }
                    }
                    closedir(droot);

                    if (found_subdirs) {
                        // iterate subdirs and label by subdir name (cat -> 1, dog -> 0 by default)
                        DIR *d = opendir(data_dir);
                        if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                        struct dirent *ent;
                        while ((ent = readdir(d))) {
                            if (ent->d_name[0] == '.') continue;
                            char subpath[4096]; snprintf(subpath, sizeof(subpath), "%s/%s", data_dir, ent->d_name);
                            struct stat st3; if (stat(subpath, &st3) != 0) continue;
                            if (!S_ISDIR(st3.st_mode)) continue;
                            // decide label by name
                            int lab = 0;
                            if (strcasestr(ent->d_name, "cat")) lab = 1; else if (strcasestr(ent->d_name, "dog")) lab = 0; else lab = 0;
                            DIR *sd = opendir(subpath);
                            if (!sd) continue;
                            struct dirent *fent;
                            while ((fent = readdir(sd))) {
                                if (fent->d_name[0] == '.') continue;
                                if (!has_ext_fn(fent->d_name)) continue;
                                char filepath[4096]; snprintf(filepath, sizeof(filepath), "%s/%s", subpath, fent->d_name);
                                Image *im = image_load_file(filepath, W, H);
                                if (!im) continue;
                                if (max_samples > 0 && count >= max_samples) { image_free(im); closedir(sd); break; }
                                if (count >= HARD_CAP) { image_free(im); closedir(sd); break; }
                                grow_fn(); images[count] = im; labels[count] = lab; count++;
                            }
                            closedir(sd);
                            if (max_samples > 0 && count >= max_samples) break;
                            if (count >= HARD_CAP) break;
                        }
                        closedir(d);
                    } else {
                        // directory contains images directly; label by filename (contains 'cat')
                        DIR *d = opendir(data_dir);
                        if (!d) { fprintf(stderr, "failed to open dir %s: %s\n", data_dir, strerror(errno)); return 1; }
                        struct dirent *ent2;
                        while ((ent2 = readdir(d))) {
                            if (ent2->d_name[0] == '.') continue;
                            if (!has_ext_fn(ent2->d_name)) continue;
                            char path2[4096]; snprintf(path2, sizeof(path2), "%s/%s", data_dir, ent2->d_name);
                            Image *im = image_load_file(path2, W, H);
                            if (!im) continue;
                            int lab = strcasestr(ent2->d_name, "cat") ? 1 : 0;
                            if (max_samples > 0 && count >= max_samples) { image_free(im); break; }
                            if (count >= HARD_CAP) { image_free(im); break; }
                            grow_fn(); images[count] = im; labels[count] = lab; count++;
                        }
                        closedir(d);
                    }
                } else {
                    // single file
                    Image *im = image_load_file(data_dir, W, H);
                    if (!im) { fprintf(stderr, "failed to load image %s\n", data_dir); return 1; }
                    grow_fn(); images[count] = im; labels[count] = 0; count++;
                }

                if (count == 0) { fprintf(stderr, "no images loaded from %s\n", data_dir); return 1; }
            }
