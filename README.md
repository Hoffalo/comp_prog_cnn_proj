# comp_prog_cnn_proj — minimal CNN in C (synthetic cat vs dog)

This project contains a minimal convolutional neural network implemented in plain C. It trains on a tiny synthetic dataset (generated patterns) to demonstrate the end-to-end pipeline: data, model, training loop, and model saving.

What is included
- `src/` — C source for a tiny CNN, utils, synthetic image generator, and a training runner (`main.c`).
- `Makefile` — build the trainer.

Quick start

Build with CMake (recommended):

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
# binary will be placed in ../bin/train
./bin/train
```

Alternatively the old Makefile is still present; you can still run `make`.

Using real images (optional)

This project supports loading images from files. By default a small PGM loader is used as a fallback. To enable `stb_image` (recommended for PNG/JPEG), do the following:

1. Put `stb_image.h` into `third_party/`.
2. Configure CMake with `-DUSE_STB=ON`:

```bash
mkdir -p build && cd build
cmake -DUSE_STB=ON ..
cmake --build .
```

3. Run the trainer and point to a dataset directory (image files containing `cat` or `dog` in filename will be labeled automatically for this demo):

```bash
./bin/train --data-dir ../my_dataset
```

If `stb_image` is not provided, the trainer will still build and you can use PGM images or the built-in synthetic dataset.

Note: this repository includes a small placeholder at `third_party/stb_image.h` so the project compiles out-of-the-box.
To enable real image loading (PNG/JPEG) replace it with the official single-file header:

```bash
mkdir -p third_party
curl -L -o third_party/stb_image.h \
	https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
```

Then configure with `-DUSE_STB=ON` and rebuild.

Notes and next steps
- This example uses synthetic 16x16 grayscale images (circle vs stripe) as proxies for cats and dogs. Replace `src/image.c` loader with a real image loader (for example `stb_image.h`) to train on real data.
- The CNN implementation is intentionally small and educational. For production or larger datasets, use optimized libraries (TensorFlow, PyTorch) or portable C libraries.
# comp_prog_cnn_proj
CNN Classification model to distinguish cats and dogs
