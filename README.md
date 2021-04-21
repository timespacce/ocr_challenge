OCR Challenge

#### Data preparation

1. 34318 train and 5509 test images with different resolutions.
2. Resize to 128x64 and normalize to [0, 1]

#### Model

1. 3 x Convolution Blocks (CONV / BN / L RELU)
2. Classification to (6, 37) (label length,  vocabulary length)
3. Softmax

#### Evaluation

1. Calculate how many pictures are correctly predicted.