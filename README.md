###OCR Challenge with Convolutional Neural Network

### Data preparation

1. 34318 train and 5509 test images with different resolutions.
2. Resize to 64x128, normalize each channel to [0, 1] and keep HWC format.

### Model

1. 3 x Convolution Blocks (**CONV** - **CONV** - **CONV** - **BATCH NORM** - **LEAKY RELU** - **DROPOUT**)
2. **CONV**(label_length)
3. **Dense**(hidden_Size)
4. **Dense**(vocab_size)
5. **Softmax**(-1)

### Training

1. Adaptive Momentum Estimation with 1e-3 and Constant Scheduling Strategy
2. batch_size = 512 on DGX A100 for 20 epochs (250 seconds)
3. Reshuffle at each epoch
4. Minimize Categorical Cross Entropy between one-hot-encoded network output and one-hot-encoded label

### Evaluation

1. Accuracy defined as number of correctly predicted pictures.
   
    | Model | Parameters : [MB] |  Train  | Test |
    | --- | --- | --- | --- |
    | 32-512 | 4,039,403 : 15.4MB | 83.9 % | 65.7 %  |

