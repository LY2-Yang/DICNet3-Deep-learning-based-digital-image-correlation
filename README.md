### README for DICNet3+ Model

#### Overview:

This paper proposes a deep learning-based digital speckle image large deformation measurement method DICNet3+, which aims to solve the "bad pixel" problem caused by unstable boundary calculation and speckle pattern tearing in complex large deformation tests. DICNet3+ introduces deep separable convolution and convolutional attention module (CBAM) to effectively reduce model parameters while improving feature extraction capability and prediction accuracy. To ensure the diversity and authenticity of the dataset, this paper combines real speckle patterns and simulated displacement fields to construct a comprehensive dataset. Although the model is only trained on the self-built dataset, the experimental results show that DICNet3+ has better displacement field prediction accuracy and generalization ability than the existing Deep-DIC method while keeping the number of parameters to a minimum. Especially in complex large deformation scenarios, it can effectively avoid the limitations of traditional DIC methods. Compared with commercial DIC software, the displacement prediction results of DICNet3+ in real experiments are basically consistent, and it has stronger robustness, which is suitable for industrial displacement measurement tasks with limited computing resources.

------

### Key Features:

1. **Depthwise Separable Convolutions**: Improves computational efficiency by performing depthwise convolutions followed by pointwise convolutions, reducing the number of parameters.
2. **CBAM (Convolutional Block Attention Module)**: Enhances the model's ability to capture spatial and channel-wise information through attention mechanisms.
3. **Group Normalization**: Replaces Batch Normalization for better generalization in small batch sizes.
4. **Multi-scale Decoder Architecture**: The model's decoder aggregates multi-scale feature maps, improving prediction accuracy in displacement fields.

------

### Prerequisites:

- **Python 3.8**
- **PyTorch 2.0**
- **Torchvision**
- **Numpy**
- **Matplotlib** (for visualization)
- **CUDA**

------

### File Structure:

- `DICNet3.py`: Defining the main DICNet3+ model architecture.

- `README.md`: This file.

- `main.py`:This script is used to predict the displacement field.

- `Simulated displacement field`:Defining the generation of the simulation displacement field.

- **Data Structure**: The dataset should contain images with references and deformations and corresponding displacement fields in the format `.npy` for each image pair.

- **Dataset address**：https://pan.baidu.com/s/1oUESoyS3__4ATVXNg_NIPQ  password：LY22
