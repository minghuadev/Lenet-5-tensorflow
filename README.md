# Lenet-5

**Lenet-5 architecture**

![image](https://github.com/wingdi/Lenet-5/blob/master/src/imgs/lenet-5.png)

**How to train:**
replace local dir of mnist in pre_data.py,then run train_and_evaluate.py

**Reference:**
https://github.com/sujaybabruwad/LeNet-in-Tensorflow

# Architecture from the above reference

Specs

Input Image is 28x28x1 and converted to 32x32x1 as per LeNet requirements.

Convolution layer 1: The output shape should be 28x28x6.

Activation 1: Your choice of activation function.

Pooling layer 1: The output shape should be 14x14x6.

Convolution layer 2: The output shape should be 10x10x16.

Activation 2: Your choice of activation function.

Pooling layer 2: The output shape should be 5x5x16.

Flatten layer: Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

Fully connected layer 1: This should have 120 outputs.

Activation 3: Your choice of activation function.

Fully connected layer 2: This should have 10 outputs.

Return the result of the 2nd fully connected layer from the LeNet function.
