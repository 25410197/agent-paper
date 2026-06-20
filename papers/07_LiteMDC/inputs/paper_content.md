# PAPER CONTENT: 07- Advancing real-time plant disease detection

**Source PDF:** papers/07- Advancing real-time plant disease detection/inputs/1-s2.0-S2772375524000133-main.pdf
**Extracted:** 2026-06-18

---

                                                             Smart Agricultural Technology 7 (2024) 100408
                                                             Contents lists available at ScienceDirect

                                           Smart Agricultural Technology

                                 journal homepage: www.journals.elsevier.com/smart-agricultural-technology

Advancing real-time plant disease detection: A lightweight deep learning

approach and novel dataset for pigeon pea crop

Sandesh Bhagat a,, Manesh Kokare a, Vineet Haswani a, Praful Hambarde b, Trupti Taori a,e,
P.H. Ghante d, D.K. Patil c

a Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded, (MS), India
b Indian Institute of Technology, Ropar, India
c Agriculture Research Station, Badnapur, India
d Indian Council of Agricultural Research, India
e Ajeenkya DY Patil University, Pune, India

ARTICLE INFO                     ABSTRACT

Editor: Stephen Symons           Plant disease detection and early disease treatment are essential for sustainable crop production. Computer vision
                                 for crop science is overgrowing with the advancement in deep learning. Real time plant disease detection poses a
Keywords:                        challenge due to the unpredictable spread of diseases within the plant, environmental factors, and the scarcity of
Computer vision                  real field datasets. The proposed work systematically addresses these issues through three key components:
Diseases detection               (a) Collaboratively generating the novel pigeon pea image dataset from agricultural fields, in partnership
Depthwise separable convolution  with 20 Agricultural Research Centers (ARS) and governmental agencies spanning 18 Indian states. (b) The
Efficient CNN                    design of lightweight and high-performance models for real-time plant disease detection in resource-constrained
                                 devices. (c) The extraction of multiscale feature of plant diseases using Multi-kernel Depthwise separable
                                 Convolutions. The proposed lightweight Lite-MDC architecture uses the Multi-kernel Depthwise separable
                                 Convolutions (MDsConv). The MDsConv module captures spatial features across various scales while maintaining
                                 a lightweight design. It effectively extracts multi-scale information to characterize plant diseases, accommodating
                                 their diverse scale. Proposed architectural approach significantly reduces computational complexity, employing
                                 only 2.2 million parameters, which is a 62% reduction compared to the standard VGG16 architecture. The
                                 proposed method outperforms the state-of-the-art networks such as InceptionV3, VGG16, ResNet50, DenseNet,
                                 MobileNet, MobileNetV3, NASNet, and EfficieNetB0 on the proposed pigeon pea dataset with 94.14% accuracy.
                                 Notably, the method achieves a 34 Frames Per Second (FPS) inference on an NVIDIA P100 GPU. Furthermore,
                                 its performance is validated across publicly available datasets, including the plant village dataset, Cassava, and
                                 apple leaf datasets, yielding 99.78%, 86.4%, and 97.2% accuracy, respectively. The Lite-MDC model exhibits the
                                 potential for real-time plant disease detection on resource-constrained edge devices such as Agriculture robots
                                 and drones.

1. Introduction                                                            and experts depend on visually inspecting plants to spot diseases. This
                                                                           method has limitations. Subjective interpretations and visual fatigue
    Globally, pigeon pea is the sixth most significant legume in Asian     can result in inaccurate assessments [4]. When diseases are misdiag-
countries [1]. Pigeon pea is a rich source of protein, high in fiber, and  nosed, it can disrupt the monitoring and control procedures, resulting
contains essential amino acids. However, the numerous occurrences of       in the improper application of pesticides, diminished plant quality, and
plant diseases may significantly restrict the plants' health and affect    reduced yields [5]. Considering the potential repercussions of misiden-
the yield. Annually, plant diseases cause production loss between 12-      tifying plant diseases, it becomes evident that precise disease identifica-
23 million dollars [2] [3]. Presently, the detection of numerous plant     tion is indispensable for preserving plant health and attaining optimal
diseases predominantly relies on human visual assessment. Farmers          yields.

 * Corresponding author.
     E-mail address: 2018pec901@sggs.ac.in (S. Bhagat).

https://doi.org/10.1016/j.atech.2024.100408
Received 6 August 2023; Received in revised form 15 January 2024; Accepted 25 January 2024
Available online 1 February 2024
2772-3755/© 2024 The Author(s). Published by Elsevier B.V. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

    Deep learning serves as an advanced tool in the realm of plant            2. Related work
disease detection, enabling the automated identification of disease-
related features [6]. However it comes with its own set of challenges             In recent years, the agricultural sector has witnessed a growing in-
like (a) The availability of real field datasets. (b) A Randomly dis-         terest in the detection and management of plant diseases, driven by
tributed plant diseases on leaves, stems, and roots, and their diversity      image processing techniques and advancements in machine learning
in texture, size, and shape. (c) Environmental challenges include image       techniques.
blur due to wind, leaf overlapping, complex background, and view-
points, as shown in Fig. 1. (d) The development of lightweight yet            2.1. Image processing and machine learning for plant diseases detection
high-performance models tailored for real-time plant disease detection
on resource-constrained embedded devices poses another significant                Image processing and machine learning techniques are extensively
challenge. Convolutional neural networks (CNNs) have emerged as a             applied for the detection of plant diseases in a variety of crops,
promising solution to tackle the challenges of disease identification         including tomatoes, potatoes, sugarcane, maize, and numerous others.
and classification. CNN architectures are specifically designed for im-       These methods encompass a range of image processing-based approaches,
age analysis and can help to overcome the limitations of traditional          such as hand-engineered features, scale-invariant feature transforms,
visual inspection methods. However, these architectures are computa-          histogram-based methods, thresholding techniques, and segmentation methods.
tionally expensive. For example, Standard VGG16 architecture has 92
layers and requires 140 M parameters for implementation [7], [8]. To
address this limitation, we proposed Lite-MDC architecture with multi-
kernel depthwise. One of the advantages of proposed architecture is its
ability to maintain state-of-the-art performance while reducing com-
putational complexity. This technique optimizes the model's efficiency
without sacrificing accuracy. This paper also proposes the First Real
field pigeon pea image dataset. The key contributions of the proposed
research work are summarized as follows:

 1. Proposed the first novel pigeon-pea disease image dataset includes
      high resolution images collected from 18 geographical locations
      across India.

 2. Proposed MDsConv block with 3x3 and 5x5 helps to extract the
      multiscale feature of plant diseases.

 3. Proposed a Light weight architecture Lite-MDC for pigeon pea dis-
      ease detection reduces computational complexity and shows state-
      of-the-art results with only 2.2 M parameters with 34 FPS suitable
      for real time application.

    To evaluate the effectiveness of the proposed approach, extensive
experiments were conducted to compare against public datasets and
existing state-of-the-art methods such as VGG-16 [9], Xceptions [10],
NasNet mobile [11], MobileNet [12], MobileNetV2 [13], ResNet50
[14], InceptionV3 [15] and DenseNet121 [16] and EfficientNetB0 [17].

    The rest of this paper is structured in the following manner. Section 2
reviews the literature work. Section 3 proposes the pigeonpea dataset
construction details; Section 4 introduces the proposed method for plant
disease detection. Section 5 enumerates the experimental details, Section
6 illustrates the results and analysis with discussion. Finally, Section 7
concludes the work with mention of some future direction.

2.2. Deep learning for plant diseases detection

    The progress made in deep learning techniques for image classification,
segmentation, and object detection is transforming the respective field
significantly. Mohanty et al. [5] used a plant village dataset and transfer
learning approach with deep CNN architectures AlexNet and GoogLeNet.
However AlexNet and GoogLeNet are computationally heavy architectures
and not suitable for edge device applications. Sladojevic et al. [30] use
CaffeNet with deep CNN architecture achieving 96.3% across 13 distinct
plant disease categories. Zhang et al. [31] use transfer learning using
AlexNet, GoogleNet, and ResNet with maximum accuracy of 97.28% for
identification of tomato leaf disease. In [33], YOLOV3 architecture (64 M
parameters) for detecting tomato diseases and pests. Fine-tuned VGG16
achieve 90.4% accuracy with 140 M parameter for apple black rot
disease severity [34]. VGG19 for potato leaves with 97.8% accuracy [35].

2.3. Efficient deep neural networks

    Real-time applications on resource-constrained devices demand
lightweight CNN architectures. Several researchers focused on the
development of efficient CNN architectures for image classification.
Iandola et al. [37] proposed SqueezeNet with 1.25 million parameters.
MobileNet [12] has been proposed for image classification as a lightweight
CNN architecture for edge device applications. Sandler et al. [13]
presents MobileNetV2 with 4 M parameters. Zoph et al. developed
NASNetMobile. The proposed study presents Lite-MDC, a lightweight
multi-kernel depth-wise convolutional network inspired by VGG16 and
multi-kernel depthwise separable convolutions, suitable for real-world
applications.

3. Datasets used

3.1. Pigeon-pea dataset construction

    The proposed work introduces the first novel pigeon-pea dataset.
Construction involves three steps: (1) System Establishment, (2) Dataset
collection and annotation, (3) Dataset processing and verification.

3.1.1. System establishment
    Research collaboration MOU signed between Technical institute and
Agriculture university. With help of ICAR and IIPR, twenty Agriculture
Research Stations (ARS) in eighteen states of India were selected.

3.1.2. Data collection and annotation
    Each research station collected images for four common pigeon-pea
diseases: Wilt, Sterility mosaic, Dry root rot, Phytophthora blight, and
healthy samples. Images captured using smartphone cameras and digital
cameras (samsung M21, Sony DSLR etc.) with multiple-resolution cameras.
All images hand-labeled by agriculture scientists.

Table 1: Data collection from Agriculture Research Stations (ARS)

Sr.No | Location | No. of Images
1. | Agriculture Research Station, Badnapur, Maharashtra | 656 / 600
2. | Agriculture Research Station, Ludhiyana, Punjab | 820 / 745
3. | Institute of Agricultural Science, BHU, Varanasi, UP | 786 / 623
4. | Agriculture Research Station, Bangaluru, Karnataka | 513 / 704
5. | Agriculture Research Station, Coimbatore, Tamil Nadu | 565 / 790
6. | Agriculture Research Station, Warangal, Telangana | 889 / 827
7. | Agriculture Research Station, Tirupati, Andhra Pradesh | 640 / 700
8. | Agriculture Research Station, Guntur, Andhra Pradesh | 875 / 587
9. | Agriculture Research Station, Dharwad, Karnataka | 348 / 255
10. | Agriculture Research Station, Kanpur, Uttar Pradesh | 336 / 369
11. | Agriculture Research Station, Khargone, Madhya Pradesh | -
12. | Agriculture Research Station, Vijayapura, Karnataka | -
13. | Agriculture Research Station, Junagadh, Gujarat | -
14. | Pulses Research Station, Sardar Krushinagar, India | -
15. | ICRISAT, Patancheru, India | -
16. | Agriculture Research Station Rahuri, India | -
17. | Indian Institute of Pulses Research, Kanpur, India | -
18. | Pulses Research Station, Sardar Krushinagar, India | -
19. | Pulses Research Station, Junagadh, Gujarat, India | -
20. | Birsa Agricultural University, Ranchi | -
Total: 12,232

Table 2: Class-wise statistics for proposed Pigeon-pea Dataset

Classes             | Collection condition | Number of images | Resolution   | Annotations type
Dry root rot        | Real field images    | 2103             | 1024 x 1024  | Image level
Healthy             | Real field images    | 2560             | 1024 x 1024  | Image level
Phytophthora blight | Real field images    | 1691             | 1024 x 1024  | Image level
Sterility Mosaic    | Real field images    | 3652             | 1024 x 1024  | Image level
Wilt                | Real field images    | 2226             | 1024 x 1024  | Image level
Total               |                      | 12,232           |              |

3.1.3. Dataset processing and verification
    Undesired blurry images eliminated. Images uniformly cropped to
1024x1024 spatial resolution. Each image cross-verified by computer
vision experts. All images resized using bilinear interpolation to
224x224x3 before being applied to CNN model. Data normalization:
subtraction of mean of each channel (RGB) from each pixel.
Dataset: 12,232 images, five classes.
GitHub: https://github.com/sandesh001/First-Pigeon-pea-datset-for-diseases-detection

3.2. Publicly available datasets

Table 3: Statistics of existing plant diseases dataset

Dataset                          | Images  | Classes | Collection    | Diseases on  | Status
Leaflet Cassava Dataset [44]     | 1,896   | 6       | Field images  | Only Leaf    | Not available
Kaggle Cassava Dataset [42]      | 9,436   | 5       | Field images  | Only leaf    | Publicly available
Leaf Diseases Dataset [30]       | 4,483   | 15      | Field images  | Only Leaf    | Not available
Plant Village Dataset [6]        | 54,309  | 38      | Lab Images    | Only Leaf    | Publicly available
Apple Foliar Disease Dataset [43]| 3,651   | 4       | Field images  | Only Leaf    | Publicly available
Apple Leaf Disease [45]          | 404     | 3       | Field images  | Only Leaf    | Not available
Crop Pest Dataset [46]           | 4,483   | 40      | Field images  | Only Leaf    | Not available
PDD271 [47]                      | 220,592 | 271     | Field images  | Only Leaf    | Not available
Pigeon-pea Dataset (Proposed)    | 12,232  | 5       | Field images  | Entire Plant | Publicly available

Table 4: Details of datasets used for experimentation

Dataset                          | Collection condition           | Resolution   | Total Images | Classes
Kaggle Cassava Dataset [42]      | Real field images              | 4000x2672    | 9,436        | 5
Apple Foliar Disease Dataset [43]| Real field images              | 1048x1365    | 3,651        | 4
PlantVillage Dataset [6]         | Indoor controlled environment  | 256x256      | 54,309       | 38
Pigeon-pea Dataset (proposed)    | Real field images              | 1024x1024    | 12,323       | 5
Total                            |                                |              | 94,700       |

3.2.1. Cassava dataset
    9,436 images, 5 classes (CGM, CMD, CBB, CBSD, Healthy).
    Resolution 4000x2672. From "Cassava leaf disease competition" on Kaggle.
    Split: 60% train / 20% val / 20% test.

3.2.2. PlantVillage dataset
    54,309 RGB images, 256x256 pixels, 38 distinct plant disease categories.
    Acquired under controlled indoor environmental conditions with uniform background.
    Split: 60% train / 20% val / 20% test.

3.2.3. Apple leaf dataset
    3,651 RGB images, resolution 1048x1365. Four classes: apple scab, cedar apple
    rust, multiple diseases, and healthy leaves. Captured with Canon Rebel T5i DSLR
    and smartphones. From Plant Pathology Challenge on Kaggle.
    Split: 60% train / 20% val / 20% test.

4. Proposed methodology

4.1. Convolutions in deep learning

    Standard convolution computational cost (Eq. 1):
    Cost_Std = Kh x Kw x C x N x H x W

    Depthwise separable convolution (DWSC) cost (Eq. 2):
    Cost_DW = Kh x Kw x C x H x W  +  C x N x H x W
    Param_DW = (Kh x Kw x C + C) + (C x N + N)

    Where Kh x Kw = Kernel size, C = Number of input channels, N = Number of
    output channels, H x W = Spatial Height and Width of input image or feature
    map. (padding = same, stride = 0, H = W assumed for simplicity.)

4.2. Proposed multi kernel depthwise separable convolution (MDsConv)

    MDsConv uses multiple kernels in depthwise convolution to extract multiscale
    features. Group size G = C / K where C = input channels, K = number of unique
    kernels. When K=1, MDConv is equivalent to standard depthwise convolution.

    Proposed MDsConv uses 3x3 AND 5x5 kernels -> G = 2.
    Input feature map is split into two groups: I = [C/2, C/2]
    - First C/2 channels -> 3x3 DwConv
    - Second C/2 channels -> 5x5 DwConv
    Outputs concatenated -> Pointwise convolution (1x1) to fuse and capture
    cross-channel dependencies.

    MDsConv total computational cost (Eq. 4):
    Cost_MDsConv = Cost_DW_3x3 + Cost_DW_5x5 + Cost_PW
    Cost_MDsConv = (3x3 x C/2 x H x W) + (5x5 x C/2 x H x W) + (C x N x H x W)
    Param_MDsConv = (3x3 x C/2 + C/2) + (5x5 x C/2 + C/2) + (C x N + N)

    Algorithm 1 (MDsConv pseudocode):
    Input: Input tensor (X), Kernel sizes [K1, K2]
    Output: Output tensor Y
    1  G = len(Kernels)
    2  Kernels = [K1, K2]
    3  Split = split(X, G, axis=-1)
    4  Out = []
    5  # Multiscale depthwise convolution
    6  for Xi, Ki in (Split, Kernels) do
    7      Yi = DepthwiseConv2D(kernel_size=Ki)(Xi)
    8      Out.append(Yi)
    9  Y = Concatenate(Out)
    10 Y = BatchNorm2D(Y)
    11 # Pointwise convolution on concatenated output
    12 Y = Conv2D(Y, kernel_size=1)
    13 Y = BatchNorm2D(Y)

4.3. Proposed Lite-MDC architecture

    Lite-MDC uses standard VGG16 as baseline. All normal convolution layers
    replaced with MDsConv (Multi kernel Depthwise separable Convolution).
    Input: 224x224x3.
    All layers followed by Batch Normalization.
    Activation: LeakyReLU after each BN layer EXCEPT the last one in the network.
    Stride of 2 used to reduce spatial dimension (in place of max pooling).
    Fully connected layers substituted with 1x1 pointwise convolution (512 filters).
    Final: Global Average Pooling -> Softmax -> 5 classes.
    Dropout: increasing rate from 0.05 to 0.4 in the last few layers.

Table 5: Proposed Lite-MDC architecture (AUTHORITATIVE from paper text)

Layer Name             | Filters size           | Stride | Output shape
-----------------------|------------------------|--------|------------------
Input Image            | -                      | -      | 224 x 224 x 3
Conv0                  | 3x3 x 3 x 64           | 1      | 224 x 224 x 64
MDConv_1               | 3x3x32, 5x5x32         | 2      | 112 x 112 x 64
PConv_1                | 1x1 x 64 x 64          | -      | 112 x 112 x 64
MDConv_2               | 3x3x32, 5x5x32         | 1      | 112 x 112 x 64
PConv_2                | 1x1 x 128 x 128        | -      | 112 x 112 x 128
MDConv_3               | 3x3x64, 5x5x64         | 2      | 56 x 56 x 128
PConv_3                | 1x1 x 128 x 128        | -      | 56 x 56 x 128
MDConv_4               | 3x3x64, 5x5x64         | 1      | 56 x 56 x 128
PConv_4                | 1x1 x 128 x 256        | -      | 56 x 56 x 256
MDConv_5               | 3x3x128, 5x5x128       | 1      | 56 x 56 x 256
PConv_5                | 1x1 x 256 x 256        | -      | 56 x 56 x 256
MDConv_6               | 3x3x128, 5x5x128       | 2      | 28 x 28 x 256
PConv_6                | 1x1 x 256 x 256        | -      | 28 x 28 x 256
MDConv_7               | 3x3x128, 5x5x128       | 1      | 28 x 28 x 256
PConv_7                | 1x1 x 256 x 512        | -      | 28 x 28 x 512
MDConv_8               | 3x3x256, 5x5x256       | 1      | 28 x 28 x 512
PConv_8                | 1x1 x 512 x 512        | -      | 28 x 28 x 512
MDConv_9               | 3x3x256, 5x5x256       | 2      | 14 x 14 x 512
PConv_9                | 1x1 x 512 x 512        | -      | 14 x 14 x 512
MDConv_10              | 3x3x256, 5x5x256       | 1      | 14 x 14 x 512
PConv_10               | 1x1 x 512 x 512        | -      | 14 x 14 x 512
MDConv_11              | 3x3x256, 5x5x256       | 1      | 14 x 14 x 512
PConv_11               | 1x1 x 512 x 512        | -      | 14 x 14 x 512
MDConv_12              | 3x3x256, 5x5x256       | 2      | 7 x 7 x 512
PConv_12               | 1x1 x 512 x 512        | -      | 7 x 7 x 512
DwConv_13              | 7x7 x 512              | 1      | 1 x 1 x 512
PConv_13               | 1x1 x 512 x 512        | -      | 1 x 1 x 512
Conv_14                | 1x1 x 512 x 512        | 1      | 1 x 1 x 512
Global Average Pooling | 1x1                    | -      | 512
Softmax                | Classifier             | -      | 5

    NOTE on PConv filter sizes interpretation: The "Filters size" column for PConv
    layers in Table 5 lists e.g. "1x1 x 64 x 64" which means input_channels x
    output_channels for the pointwise (1x1) convolution. For MDConv layers,
    "3x3x32, 5x5x32" means 32 filters of 3x3 kernel and 32 filters of 5x5 kernel
    (total 64 output channels before pointwise).

    Total: 1167 M additions and multiplications, 2.2 M parameters.

4.4. Computational complexity calculations for Lite-MDC

    Conv0 layer (3x3, 64 filters, input 224x224x3):
    Mul_Conv0 = 224 x 224 x 3 x 3 x 3 x 64
    Add_Conv0 = 224 x 224 x (3x3x3 - 1) x 64
    Param_Conv0 = 3 x 64 + 64

    MDConv_1 layer (112x112x64 input, stride=2):
    Mul_MDConv1 = H x W x C/2 x (K1xK1 + K2xK2)
               = 112 x 112 x 32 x (3x3 + 5x5)
    Add_MDConv1 = H x W x C/2 x {(K1xK1-1) + (K2xK2-1)}
               = 112 x 112 x 32 x {(3x3-1) + (5x5-1)}
    Param_MDConv1 = (3x3 x 32 + 32) + (5x5 x 32 + 32)

    PConv_1 layer (112x112x64, output 64):
    Mul_PConv1 = H x W x C x N = 112 x 112 x 64 x 64
    Add_PConv1 = H x W x (C-1) x N = 112 x 112 x (64-1) x 64
    Param_PConv1 = C x N + N = 64 x 64 + 64

5. Experimentation

5.1. Evaluation metrics

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN + FP + FN) / (TP + TN + FP + FN)   [NOTE: denominator should be sum]
    F1 Score = 2 x (Precision x Recall) / (Precision + Recall)

5.2. Training and implementation

    Framework: Keras with TensorFlow backend [49]
    Hardware: Ubuntu 16.04 OS, Intel 2.54 GHz Processor, 16 GB RAM, NVIDIA Tesla P100 GPU
    Batch size: 16
    Optimizer: Adam [50]
    Loss function: categorical cross-entropy
    Learning rate schedule: if validation loss constant for 5 successive epochs -> reduce by factor 0.3
    Input image size: 224 x 224 x 3
    Dataset split: 60% train / 20% val / 20% test (for all datasets)
    Training augmentation: rotation (range=360), width shift (0.2), height shift (0.2),
    shear (0.2), zoom (0.2), brightness (0.1-1.0), horizontal flip, vertical flip, nearest fill mode.
    Number of input images increased by 2x with augmentation.

6. Results and discussion

6.1. Comparison with standard convolution

Table 6: Comparison StdConv vs MDsConv (Input 224x224x16, Stride=1, Padding=0, Output channels=64)

Method  | Input          | Parameter | Mul_Add   | Kernels
--------|----------------|-----------|-----------|--------
StdConv | N=64, C=16     | 9,280     | 921.6 M   | 3x3
MDsConv | N=64, C=16     | 1,376     | 14.20 M   | 3x3, 5x5

MDsConv: 7x fewer parameters, 65x less computational cost than standard convolution.

6.2. Computational complexity comparison

Total for Lite-MDC: 1167 M computations, 2.2 M parameters.
VGG16: 15,300 M computations, 140 M parameters.

Table 11: Comparison Lite-MDC vs VGG16

Method               | Computations | Accuracy | Total Parameter
---------------------|--------------|----------|-----------------
VGG16 [9]            | 15,300 M     | 87.33%   | 140 M
Lite-MDC (Proposed)  | 1,167 M      | 95.34%   | 2.2 M

6.3. Comparison with state-of-the-art methods on pigeon-pea dataset

Table 9: Comparison with SOTA for pigeon-pea detection

Model               | Parameters | Accuracy | FPS | FLOPS
--------------------|------------|----------|-----|--------
Standard VGG16      | 140 M      | 87.33    | 5   | 31.1 G
ResNet50            | 25.63 M    | 92.30    | 23  | 7.85 G
InceptionV3         | 23.85 M    | 79.34    | 7   | 6.10 G
Xception            | 22.91 M    | 84.90    | 23  | 9.13 G
DenseNet121         | 7.0 M      | 89.56    | 9   | 7.04 G
EfficientNetB0      | 5.33 M     | 83.89    | 26  | 0.89 G
NASNet mobile       | 5.3 M      | 88.30    | 10  | 1.15 G
MobileNet           | 4.2 M      | 86.63    | 28  | 1.15 G
MobileNetV2         | 3.5 M      | 86.37    | 8   | 0.93 G
Lite-MDC (Proposed) | 2.2 M      | 94.14    | 33  | 0.61 G

Table 7: Comparison of proposed method Lite-MDC on different datasets

Datasets                          | Precision | Recall | F1 Score | Accuracy (%) | Parameters
----------------------------------|-----------|--------|----------|--------------|------------
Cassava Dataset [42]              | 86        | 90     | 91       | 87.20%       | 2.2 M
Apple Foliar Disease Dataset [43] | 86        | 90     | 91       | 97.36%       | 2.2 M
PlantVillage Dataset [6]          | 99        | 99     | 99       | 99.77%       | 2.2 M
Pigeon-pea Dataset (proposed)     | 94        | 94     | 95       | 95.01%       | 2.2 M

Table 8: Class-wise accuracy on Pigeon-pea, Cassava, and Apple leaf datasets

PIGEON-PEA (Proposed):
Class                  | Total Samples | Right Pred | Wrong Pred | Precision | Recall | F1-Score | Accuracy%
Dry root rot (DRR)     | (not listed)  | (not listed)| (not listed)| 95       | 99     | 97       | 98.0
Healthy                | 220           | 217         | 3          | 92        | 90     | 91       | 90.0
Phytophthora blight(PB)| 252           | 227         | 25         | 95        | 92     | 93.0     | 92.0
Sterility mosaic (SM)  | 340           | 313         | 27         | 96        | 97     | 97.0     | 97.0
Wilt disease (WD)      | 730           | 709         | 21         | 95        | 95     | 95       | 93.7
                       | 445           | 417         | 28         |           |        |          |

CASSAVA LEAF DATASET:
Class                            | Total | Right | Wrong | Precision | Recall | F1  | Accuracy%
Cassava Bacterial Blight (CBB)   | 109   | 76    | 33    | 56        | 50     | 52  | 50.0
Cassava Brown Streak Disease(CBSD)| 219  | 171   | 48    | 83        | 78     | 80  | 78.2
Cassava Green Mottle (CGM)       | 210   | 183   | 27    | 84        | 77     | 80  | 77.0
Cassava Mosaic Disease (CMD)     | 1280  | 1259  | 21    | 95        | 96     | 95  | 96.0
Healthy                          | 216   | 188   | 28    | 95        | 95     | 95  | 93.7

APPLE LEAF DATASET:
Class              | Total | Right | Wrong | Precision | Recall | F1  | Accuracy%
Healthy leaves     | 103   | 102   | 1     | 98        | 99     | 99  | 99.00
Multiple diseases  | (not listed fully) |  |  | 56     | 67     | 61  |
Apple Rust         | 18    | 11    | 7     | 83        | 99     | 98  | 99.0
Apple Scab         | 221   | 124   | 97    | 96        | 98     | 98  | 98.3
                   | 119   | 117   | 2     | 97        |        |     |

6.4. Comparison with existing studies on public datasets

Table 10: Comparison with SOTA on PlantVillage, Cassava, Apple leaf Datasets

PLANTVILLAGE [6]:
Method                  | Parameter  | Accuracy%
AlexNet [38]            | 126 M      | 99.24%
DenseNet169 [16]        | 14.31 M    | 99.89%
GoogleNet [15]          | 23.85 M    | 99.76%
ResNet34 [14]           | 21.6 M     | 99.67%
ResNet152 [14]          | 60.42 M    | 99.69%
VGG13 [9]               | 135.6 M    | 99.49%
Squeezenet [37]         | 1.25 M     | 97.20%
Xinda Liu et al. [47]   | -          | 99.78%
Lite-MDC (Proposed)     | 2.2 M      | 99.78%

CASSAVA LEAF DATASET [42]:
Method                  | Parameter  | Accuracy%
VGG16 [9]               | 140 M      | 70.1%
MobileNet [12]          | 4.25 M     | 68.5%
ResNet50 [14]           | 25.64 M    | 81.5%
H R Ayu et al. [51]     | 3.53 M     | 65.06%
Alabdulmohsin [52]      | -          | 68.8%
Lite-MDC (Proposed)     | 2.2 M      | 86.4%

APPLE LEAF DATASET [43]:
Method                  | Parameter  | Accuracy%
AlexNet [38]            | 62.37 M    | 65.6%
VGG16 [9]               | 140 M      | 90.1%
MobileNetV2 [12]        | 3.5 M      | 85.98%
ResNetV2 [14]           | 44 M       | 93.68%
InceptionV3 [15]        | 23.85 M    | 89.28%
Manaswini et al. [53]   | -          | 73.56%
Trong et al. [54]       | 4.2 M      | 91.54%
Alsayed et al. [55]     | -          | 68.8%
Lite-MDC (Proposed)     | 2.2 M      | 97.2%

6.6. Ablation study

Table 12: Ablation study of Lite-MDC on Pigeon-pea Dataset

Methods          | DWConv in FC | DWConv | MDsConv | Dropout | Accuracy | Parameters | FLOPS
-----------------|--------------|--------|---------|---------|----------|------------|--------
VGG-16           |              |        |         |         | 87.33    | 140 M      | 31.0 G
Lite-MDC-Mod1    | yes          |        |         |         | 90.53    | 15 M       | 10.70 G
Lite-MDC-Mod2    | yes          | yes    |         |         | 91.58    | 2.2 M      | 0.60 G
Lite-MDC-Mod3    | yes          | yes    | yes     |         | 93.80    | 2.21 M     | 0.61 G
Lite-MDC (Prop.) | yes          | yes    | yes     | yes     | 94.14    | 2.21 M     | 0.61 G

7. Conclusion

    Lite-MDC: 2.2 M parameters, 94.14% accuracy on pigeon-pea dataset, 33 FPS on
    NVIDIA P100 GPU (16 GB RAM). Multi-kernel depthwise convolution with 3x3 and
    5x5 kernels improves multiscale feature extraction. 63x less parameters than
    VGG16. Results on four datasets: pigeon-pea 94.14%, Plant Village 99.78%,
    Cassava 86.4%, Apple leaf 97.2%.

---

## EXTRACTED FIGURES (visual analysis)

### Figure 7 — Convolution operations (page 7, `papers/07- Advancing real-time plant disease detection/inputs/figures/page7.png`)

Fig. 7 shows three sub-figures comparing convolution types:

**(a) Standard convolution:**
A 3D input volume (H x W x C) is convolved with N filters of kernel size K x K,
producing output H x W x N. Labels: "Standard Convolution", "N Filters".

**(b) Depthwise separable convolution:**
Two-stage operation:
- Stage 1 (Depthwise): each input channel convolved independently with one K x K kernel
  -> output H x W x C. Label: "Depthwise Convolution", "C Filters".
- Stage 2 (Pointwise): 1x1 convolution with N filters fuses C depthwise outputs
  -> H x W x N. Label: "Pointwise Convolution", "N Filters".

**(c) Multi-kernel Depthwise Separable Convolution (MDsConv):**
- Input (H x W x C) split into two groups of C/2 channels.
- Group 1 (C/2 channels): depthwise 3x3 DWConv -> H x W x C/2
- Group 2 (C/2 channels): depthwise 5x5 DWConv -> H x W x C/2
- Outputs concatenated -> H x W x C
- Pointwise convolution (1x1, N filters) -> H x W x N
Labels visible: "Multi-Kernel Depthwise Convolution" and "Pointwise Convolution",
"K1 x K1 = C/2 Filters", "K2 x K2 = C/2 Filters".

---

### Figure 8a — Proposed Lite-MDC Architecture (page 8, `papers/07- Advancing real-time plant disease detection/inputs/figures/page8_litemdc_arch_zoom.png`)

The figure shows the Lite-MDC architecture as a vertical stack (read bottom-to-top,
i.e., from Input Image to GAP+softmax). The notation used in figure labels is:
  MDsConv(K1_filters)(K2_filters), output_filters, LR/stride

Blocks visible in figure (bottom to top), exactly as labeled:

1. Input Image — 224 x 224 x 3 — x1
2. Conv(3,3,64,LR/2) [Conv0: 3x3, 64 filters, stride=1] — 224 x 224 x 64 — x1
3. MDsConv(3,32)(5,32),64,LR/2 [MDConv_1, stride=2] — 112 x 112 x 64 — x2
4. MDsConv(3,32)(5,32),64,LR/1 [MDConv_2, stride=1] — 112 x 112 x 64 — x1
5. MDsConv(3,64)(5,64),128,LR/2 [MDConv_3, stride=2] — 56 x 56 x 128 — x2
6. MDsConv(3,64)(5,64),128,LR/1 [MDConv_4, stride=1] — 56 x 56 x 128 — x1
7. MDsConv(3,128)(5,128),256,LR/2 [MDConv_6, stride=2] — 28 x 28 x 256 — x2
8. MDsConv(3,128)(5,128),256,LR/1 [MDConv_7, stride=1] — 28 x 28 x 256 — x1
9. MDsConv(3,256)(5,256),512,LR/2 [MDConv_9, stride=2] — 14 x 14 x 512 — x2
10. MDsConv(3,256)(5,256),512,LR/1 [MDConv_10, stride=1] — 14 x 14 x 512 — x1
11. MDsConv(3,256)(5,256),512,LR/1 [MDConv_11, stride=1] — 14 x 14 x 512 — x1
12. MDsConv(3,256)(5,256),512,LR/2 [MDConv_12, stride=2] — 7 x 7 x 512 — x2
13. DwConv(7),512 [DwConv_13: 7x7 depthwise, stride=1] — 1 x 1 x 512 — x1
14. 1 x 1 x 512 [PConv_13] — 1 x 1 x 512 — x1
15. GAP, softmax [Global Average Pooling + Softmax] — 5 classes — x1

NOTE: The figure diagram is consistent with Table 5. Table 5 is the authoritative
and complete source for all 26 named layers. The figure uses a condensed view
where some MDConv+PConv pairs are shown as single grouped blocks.
The "x1" and "x2" annotations in the figure indicate stride (not repetition count).

---

### Figure 8b — Proposed MDsConv Block (page 8, `papers/07- Advancing real-time plant disease detection/inputs/figures/page8_mdsconv_block_zoom.png`)

The MDsConv block diagram (high-resolution zoomed view) shows the following data
flow precisely (bottom to top):

1. Input: H x W x C (enters at bottom)
2. Channel Split: splits input into two halves:
   - Left branch output: H x W x C/2
   - Right branch output: H x W x C/2
3. Left branch: DWConv 3x3, BN -> output H x W x C/2
4. Right branch: DWConv 5x5, BN -> output H x W x C/2
5. Both branch outputs (each H x W x C/2) concatenated and fed into:
6. Conv1 x1, BN (Pointwise 1x1 convolution) -> output H x W x N
7. Output: H x W x N (exits at top)

Block colors in figure: Channel Split = orange/tan background; DWConv blocks = green;
Conv1x1+BN = pink/red; surrounding dashed box = light blue background.

IMPORTANT: No explicit LeakyReLU activation is shown INSIDE the MDsConv block
diagram itself. Per text: "LeakyReLU activation function after each batch normalization
layer except the last one in the network." So activation is applied after each BN step
in the DWConv sub-blocks, not as a separate drawn element in the figure.
