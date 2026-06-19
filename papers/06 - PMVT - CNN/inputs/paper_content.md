# PAPER CONTENT: 06 - PMVT - CNN

**Source PDF:** E:/UIT/study/claude/learn-claude/papers/06 - PMVT - CNN/inputs/fpls-14-1256773.pdf
**Extracted:** 2026-06-17

---

                                                   TYPE Methods
                                                   PUBLISHED 26 September 2023
                                                   DOI 10.3389/fpls.2023.1256773

OPEN ACCESS                                    PMVT: a lightweight vision
                                               transformer for plant disease
EDITED BY                                      identification on mobile devices

Ce Yang,                                       Guoqiang Li1, Yuchao Wang , 2,3 Qing Zhao1, Peiyan Yuan2,3
University of Minnesota Twin Cities,           and Baofang Chang 2,3*
United States
                                               1Institute of Agricultural Economics and Information, Henan Academy of Agricultural Sciences,
REVIEWED BY                                    Zhengzhou, Henan, China, 2College of Computer and Information Engineering, Henan Normal
                                               University, Xinxiang, Henan, China, 3Key Laboratory of Artificial Intelligence and Personalized Learning
Congliang Zhou,                                in Education of Henan Province, Xinxiang, Henan, China
University of Florida, United States
Lingxian Zhang,                                Due to the constraints of agricultural computing resources and the diversity of
China Agricultural University, China           plant diseases, it is challenging to achieve the desired accuracy rate while
                                               keeping the network lightweight. In this paper, we proposed a computationally
*CORRESPONDENCE                                efficient deep learning architecture based on the mobile vision transformer
                                               (MobileViT) for real-time detection of plant diseases, which we called plant-
Baofang Chang                                  based MobileViT (PMVT). Our proposed model was designed to be highly
    changbaofang@htu.edu.cn                    accurate and low-cost, making it suitable for deployment on mobile devices
                                               with limited resources. Specifically, we replaced the convolution block in
RECEIVED 11 July 2023                          MobileViT with an inverted residual structure that employs a 7x7 convolution
ACCEPTED 08 September 2023                     kernel to effectively model long-distance dependencies between different leaves
PUBLISHED 26 September 2023                    in plant disease images. Furthermore, inspired by the concept of multi-level
                                               attention in computer vision tasks, we integrated a convolutional block attention
CITATION                                       module (CBAM) into the standard ViT encoder. This integration allows the
                                               network to effectively avoid irrelevant information and focus on essential
Li G, Wang Y, Zhao Q, Yuan P and Chang B       features. The PMVT network achieves reduced parameter counts compared to
(2023) PMVT: a lightweight vision              alternative networks on various mobile devices while maintaining high accuracy
transformer for plant disease                  across different vision tasks. Extensive experiments on multiple agricultural
identification on mobile devices.              datasets, including wheat, coffee, and rice, demonstrate that the proposed
Front. Plant Sci. 14:1256773.                  method outperforms the current best lightweight and heavyweight models.
doi: 10.3389/fpls.2023.1256773                 On the wheat dataset, PMVT achieves the highest accuracy of 93.6% using
                                               approximately 0.98 million (M) parameters. This accuracy is 1.6% higher than that
COPYRIGHT                                      of MobileNetV3. Under the same parameters, PMVT achieved an accuracy of
                                               85.4% on the coffee dataset, surpassing SqueezeNet by 2.3%. Furthermore, out
2023 Li, Wang, Zhao, Yuan and Chang.           method achieved an accuracy of 93.1% on the rice dataset, surpassing
This is an open-access article distributed     MobileNetV3 by 3.4%. Additionally, we developed a plant disease diagnosis app
under the terms of the Creative Commons        and successfully used the trained PMVT model to identify plant disease in
Attribution License (CC BY). The use,          different scenarios.
distribution or reproduction in other
forums is permitted, provided the original     KEYWORDS
author(s) and the copyright owner(s) are
credited and that the original publication in  plant disease identification, vision transformer, lightweight model, attention
this journal is cited, in accordance with      module, APP
accepted academic practice. No use,
distribution or reproduction is permitted
which does not comply with these terms.

Frontiers in Plant Science                     01  frontiersin.org
Li et al.                                                                     10.3389/fpls.2023.1256773

1 Introduction                                                                paradigm to CNNs, the ViT has attracted significant attention and
                                                                              achieved considerable success in the field of computer vision (Khan
    Plant disease is one of the contributing factors to the global            et al., 2022; Lin et al., 2022).
decrease in grain production (Savary et al., 2019), and real-time
detection of plant disease has an important impact on the                         With the significant advancements of CNNs and ViT networks
agricultural industry. Applying deep learning models significantly            in plant disease recognition technology, a prevailing trend among
simplifies the entire process and enables end-to-end technical                network models is to augment the number of parameters in order to
services. Currently, there are two typical architectures for plant            enhance performance. These enhancements in performance are
disease recognition: convolutional neural network (CNN)-based                 accompanied by an increase in model size (network parameters)
architectures and vision transformer (ViT)-based architectures.               and latency (Han et al., 2021; Wu et al., 2021; Yu et al., 2022b). They
These methods extract explicit features from images and                       overlook a common issue: plant disease identification is typically
automatically perform classification, which is key for plant                  conducted on edge devices, such as smartphones and embedded
disease recognition.                                                          devices. Such devices usually have restricted computing power,
                                                                              storage capacity, and energy supply. Hence, using a lightweight
    Over the past few years, the application of CNNs to                       network can decrease the size and computational complexity of the
identifying plant diseases has gained in popularity with the                  model, thereby improving its compatibility with resource
development of artificial intelligence technology. For instance,              constraints. Numerous researchers have recently been studying
Akshai and Anitha (2021) compared various CNNs using the                      the application of affordable network models for real-time plant
PlantVillage dataset (Hughes and Salathe 2015) and reported                   disease detection. Concretely, Bao et al. (2021) proposed SimpleNet,
that the DenseNet model with feature map reuse achieved the                   which achieved 94.10% wheat recognition accuracy with only 2.13
highest accuracy of 98.27%. Another study by Yu et al. (2022a)                million (M) parameters. In addition, the apple leaf disease
used a ResNet network with a residual structure to identify apple             identification method based on the cascade backbone network
leaf diseases, and it obtained an average F1-score of 95.70%.                 (CBNet) proposed by (Sheng et al., 2022) achieved an accuracy
CNNs can efficiently extract significant features from images and             rate of 96.76%. Moreover, the VGG-ICNN model proposed by
accomplish plant disease identification automatically. The                    Thakur et al. (2023) has 6 M parameters, which is lower than most
primary reason for this is that CNNs have the characteristic of               deep learning models; and it performs well on multiple datasets
parameter sharing, which reduces the number of parameters in                  such as apple, corn, and rice. Generally, the methods mentioned
the model and addresses the overfitting issue seen in computer                above primarily concentrate on identifying a single plant disease,
vision tasks. Therefore, the application of deep learning                     while other methods exhibit imbalances in identification accuracy
technology based on CNNs has made significant progress in                     and calculation cost. Hence, to enhance the real-time performance
plant disease diagnosis (Hasan et al., 2020; Xiong et al., 2021;              of plant disease identification, it is crucial to employ a low-latency
Ahmad et al., 2022). Nonetheless, there will be an increase in                and highly accurate network model.
unnecessary computational overhead as a network's depth
increases. Simultaneously, the convolutional layer of CNNs                        Achieving high-accuracy and low-cost plant disease
only considers the characteristics of the local area during                   identification in agricultural environments with limited
convolution and does not explicitly incorporate the positional                computing resources presents a significant challenge. The
information of pixels. This will impact the effectiveness of a plant          majority of existing lightweight networks focus on a single plant
disease identification model.                                                 disease. However, when faced with numerous types of plant
                                                                              diseases, they fail to deliver satisfactory performance. In this
    To address the above issues, Dosovitskiy et al. (2020) proposed           paper, we introduced a lightweight model for plant disease
a vision transformer (ViT) architecture based on a self-attention             diagnosis based on MobileViT (Mehta and Rastegari, 2021),
mechanism (Vaswani et al., 2017) to replace the traditional CNN               which has a low computational cost and is competitive in terms
for image recognition. A ViT architecture divides an image into               of inference speed. In particular, the crisscrossing leaves in the
non-overlapping patches and applies multi-head self-attention                 agricultural dataset lead to an unsatisfactory recognition effect with
within the transformer encoder to learn representations of                    MobileViT. Thus, we consider using a larger convolution kernel (7
patches. Although this paradigm considers the global relationship             x 7) to analyze the connection between different leaves. Using larger
of images and has achieved satisfactory results in plant disease              convolution kernels allows us to model the dependencies between
recognition, it usually requires a large quantity of training data to         long-distance pixels (Liu et al., 2021; Liu et al., 2022) and enhance
achieve relatively high accuracy. Hence, alternating the use of CNNs          the ability of the model to capture global information from plant
and ViTs to extract more comprehensive features has become a                  disease images. Additionally, focusing on the salient leaf regions in
better choice in plant disease diagnosis. Take a classic case: Lu et al.      plant images can improve the robustness of the model. We used the
(2022) introduced a ghost module into the ViT encoder, which                  CBAM (Woo et al., 2018) to adjust feature weights in various
extracts different levels of features in an image. Their model                channels of the transformer encoder. Finally, we employed a
achieved an accuracy rate of 98.14% in detecting grape leaf                   residual network to fuse the initial feature map and improve the
diseases and insect pests in the field. Similarly, Yu et al. (2023)           fitting ability of the model. We named this model plant-based
used inception blocks to enhance the ability of the ViT encoder to            MobileViT (PMVT) and deployed it to identify plant diseases in
extract local information; they achieved optimal performance on               datasets and in various scenarios. Experimental results indicate that
four typical plant disease datasets. As an alternative architectural          PMVT surpasses the current leading lightweight networks and

Frontiers in Plant Science                                                02  frontiersin.org
Li et al.                                                                                              10.3389/fpls.2023.1256773

heavyweight models, thereby demonstrating its effectiveness as a           2.1.2 Coffee
versatile backbone network across various datasets.
                                                                               The coffee (Parraga-Alava et al., 2019) dataset contains three
    The main contributions of this paper are as follows.                   types of coffee leaves: healthy, red spider mite, and rust. Images of
                                                                           the same size and resolution are included in each category of
     * We used a low-cost ViT model for plant disease diagnosis.           leaves. The dataset was collected in a natural field environment,
         This model is computationally efficient and can function as       where the background of the pictures contains various
         a generic backbone network on mobile devices.                     disturbances such as weeds and soil. Since some sample features
                                                                           are not significant enough, we selected a thousand of them to build
     * We introduced a 7x7-sized convolution kernel into the               a new dataset.
         convolution block for modeling long-distance pixel-to-pixel
         dependencies. Moreover, the CBAM guides the network to            2.1.3 Rice
         learn the weights between various channels, which
         enhances the fitting ability of MobileViT to image feature            The rice (Sethy, 2020) dataset lends itself to the classical binary
         representation.                                                   classification problem as it contains samples classified simply as
                                                                           either healthy or unhealthy rice. The resolution of the images in this
     * We conducted comparative experiments on several datasets            dataset varies in size. Furthermore, some of the images in this
         obtained under different scenarios, and the results revealed      dataset have a uniform white background, which makes the dataset
         that our method not only competes with similarly sized            ideal for testing model performance in both a controlled laboratory
         lightweight networks but also outperforms state-of-the-art        environment and a real field environment.
         heavyweight networks.

2 Materials and methods                                                    2.2 Our proposed method

2.1 Datasets                                                               2.2.1 Overall structure of PMVT

    We randomly divided three datasets into a training set,                    Figure 2 depicts the overall structure of our model, which
validation set, and testing set according to the ratio of 8:1:1.           comprises five layers. Before pushing input into the block, the
Table 1 shows the details of each dataset and how many samples             feature map is downsampled using a 3 x 3 convolution; this is
comprised each subset. Figure 1 displays some samples of                   followed by an inverted residual block or a standard transformer
the datasets.                                                              encoder. The inverted residual block is used to extract local features
                                                                           of the image and capture the long-distance dependencies between
2.1.1 Wheat                                                                distant pixels. The MobileViT block uses a self-attention
                                                                           mechanism to model the global relationship of the image and
    The wheat (Lian, 2022) dataset comprises 4087 images of varying        employs a CBAM block to make up the channel attention and
sizes depicting seven different categories of wheat diseases. The          spatial attention information. The channel dimension is expanded
images include the real-world environmental factors that interfere         by four times using a 1 x 1 convolution in the last layer of the
with identifying the wheat crop, such as sky, soil, and weeds.             network to better adapt to computer classification tasks. PMVT
                                                                           contains three different network sizes: extra extra small (XXS); extra

TABLE 1 Data distributions for the datasets used in our comparative experiments.

Name                        Class  Diseases     Training set size                 Validation set size  Testing set size

Wheat                          0   health                   528                                 65                 59
                               1   rust                     673                                 83                 77
Coffee                         2   mildew                   282                                 34                 32
Rice                           3   smut                     674                                 83                 75
                               4   root rot                 381                                 46                 41
                               5   scab                     391                                 48                 45
                               6   leaf spot                378                                 47                 45
                               0   healthy red              353                                 43                 39
                               1   spider mite              136                                 16                 15
                               2   rust                     324                                 39                 35
                               0   healthy                  407                                 50                 44
                               1   unhealthy                413                                 50                 43

Frontiers in Plant Science                                             03                              frontiersin.org
Li et al.                                                                                                                            10.3389/fpls.2023.1256773

FIGURE 1

Sample images from the (A) wheat dataset, (B) coffee dataset, and (C) rice dataset.

small (XS); and small (S)). These sizes correspond to those               representation information of the image in a more blocky manner
in MobileViT.                                                             and reduce the computational cost of the self-attention mechanism.
                                                                          Through the 1 x 1 convolution kernel, the output of the transformer
2.2.2 Inverted residual block                                             is restored to the original channel dimension, and the channel
                                                                          attention and spatial attention information are learned through the
    An inverted residual block is a standard convolutional structure      CBAM block. Finally, the obtained feature map is spliced with the
comprising three convolution kernels. Before extracting image             original feature map to prevent loss of feature information and is
features, a 1 x 1 convolution kernel is used to increase the              then input to the next stage after a 3 x 3 convolution.
channel dimension, generally by two times. Then, we replace the
3 x 3 convolution kernel of the original MobileViT with a 7 x 7           2.2.4 Vision transformer encoder
convolution kernel, thus making it easier to capture long-distance
dependencies between pixels. In addition, depthwise separable             As shown in Figure 4B, the encoder used to learn image features
convolutions are used to reduce the computational complexity of
the model and increase the inference speed. Finally, we use a 1 x 1       consisting of standard transformer blocks. First, an image with
convolution kernel to restore the channel dimension of the image.
Figure 3 shows the overall structure of the inverted residual block.      dimensions [C,H,W] is divided into patches of P size, and a linear

2.2.3 Mobile ViT block                                                    transformation is applied to each patch for flattening. Positional

    As described in Figure 4A, learning global representations of         encoding information is then applied to each patch; through this,
feature maps using 1 x 1 and 3 x 3 convolutions. Before entering the
standard transformer encoder, the same color patch at the same            each patch then has dimensions of (H/Pi, W/Pi, C). Next, we use three
position is taken out and put into the same sequence for self-
attention calculation. This measure allows us to learn the global
                                                                          learnable parameter matrices to multiply each patch to get queries

                                                                          (WQ), keys (WK), and values (WV). For patch i, we apply the dot

                                                                          product to the query matrix with the key matrix of the remaining

                                                                          patches, and then we divide by the number of key matrix elements.

                                                                          Finally, we apply the softmax function to obtain the attention scores

                                                                          of the remaining patches for patch i. These attention scores are

FIGURE 2

Overview of the PMVT model. 2 means to downsample the feature map twice, and L stands for repeated stacking of L MobileViT blocks. For
computer vision classification tasks, we use a classifier composed of an average pooling layer and a fully connected layer.

Frontiers in Plant Science                                            04                                                                          frontiersin.org
Li et al.                                                                     10.3389/fpls.2023.1256773

FIGURE 3                                                                      2.2.5 CBAM block

Structure of the inverted residual block. C' represents the feature               The CBAM block is composed of a channel attention module
information obtained by convolving each channel of the feature                and a spatial attention module, and it uses a 3 x 3 convolution
map using a convolution kernel.                                               kernel to preprocess the feature map before insertion. We pass the
                                                                              input feature map through a parallel average pooling layer and max
multiplied by the value matrix of patch i to obtain the feature               pooling layer, and then we change the feature map from [C,H,W] to
information. Equation 1 illustrates the process of the entire                 [C,1,1] dimensions. The shared MLP module comprises two 1 x 1
attention mechanism. MLP comprises two fully connected layers                 convolution kernels, which compress the number of channels to R
and employs an incentive compression mechanism to learn                       times the original number and then expand it back to the original
interaction information between different dimensions.                         number of channels. The feature maps obtained by the average
                                                                              pooling layer and the max pooling layer are spliced to obtain the
                                                                              weights of each channel, which are finally multiplied by the original
                                                                              feature map. Equation 2 describes the weight assignment process of
                                                                              the channel attention module. s stands for using Sigmoid as the
                                                                              activation function, W1 in R^(C/r x C), and W1 in R^(C/r x C). W1 and W0 are
                                                                              shared weights for the two inputs of the max pooling layer and the
                                                                              average pooling layer.

                                                                              Mc(F) = s (MLP(AvgPool(F)) + MLP(MaxPool(F)))                       (2)

           self - attention = softmax ( QK^T / sqrt(dk) ) x V
                                                                    (1)
                                                                                  The output of the channel attention module is obtained through
                                                                              the max pooling layer and average pooling layer. We acquire two

                            A

                            B

                            C

FIGURE 4

Detailed description of the vision transformer block. (A) The overall structure of the vision transformer block; (B) the structure of the vision
transformer block encoder; and (C) the architecture of the CBAM block, where x represents the multiplication with the original feature map.

Frontiers in Plant Science                                                05                                         frontiersin.org
Li et al.                                                                          10.3389/fpls.2023.1256773

feature maps with dimensions of [1,H,W], and then we splice them.            select pictures from their album for identification (Figure 5C).
Through a 7 x 7 convolution, we obtain a feature map of one                  Furthermore, users have the option to search for plant diseases
channel and multiply it by the original feature map. Equation 3              based on specific conditions or criteria (Figure 5D). The application
shows the forward process of the spatial attention module, while             then presents the relevant categories of plant diseases based on the
Figure 4C shows the forward process of the entire CBAM block.                selected pictures or conditions. Figure 5E displays the final
                                                                             identification results of plant diseases and the corresponding
           Ms(F) = s (f^(7x7)([AvgPool(F); MaxPool(F)]))            (3)      control methods.

               = s (f^(7x7)([F_avg^s ; F_max^s]))                           2.4 Experimental details

2.3 App for plant disease identification                                          Data augmentation has been shown to improve model
                                                                             robustness and generalization. Before training the network, all
    We export the trained model to an open neural network                    images are uniformly resized to 224 x 224. The samples in the
exchange (ONNX) file format to preserve crucial details such as              training, validation, and test sets are randomly rotated and cropped
structure and weights. The model is converted into an NCNN file              along the center. Finally, we normalize all images using standard
format for storage to facilitate deployment on a mobile terminal for         deviation and mean square deviation. Table 2 describes our
inference because the NCNN format is a high-performance neural               hyperparameter settings for model training.
network inference framework optimized for mobile platforms.
Subsequently, the structure and weight information of the model              2.5 Model evaluation
are extracted for plant disease identification using the C++
language. The XML language is used to define the layout and                      In this study, we use top-1 accuracy (Equation 4) to determine
appearance of the application front-end interface. Lastly, the               the highest accuracy that the model can achieve. We also use
back-end interaction of the application is developed using the               precision (Equation 5) and recall (Equation 6) to evaluate the
JAVA language, while the MySQL database is used for storing                  performance of the model. Parameters, floating point operations
plant diseases and related information. As shown in Figure 5, the            per second (FLOPs), and frames per second (FPS; the number of
app possesses the capability to perform photo identification using           images the model processes per second) are used to express the
the camera of the device (Figure 5B). Alternatively, it allows users to

FIGURE 5

Introduction of plant disease identification app. (A) the main page of the app; (B) the page for camera recognition; (C) the page to select local
albums for recognition; (D) the page for disease search; and (E) the page displaying disease identification results.

TABLE 2 Hyperparameter settings for training.

Name                                           Value                            Description

Epochs                                         100                              Number of times the model was trained
Batch size                                     32                               Number of samples selected for one training
Optimizer                                      AdamW                            Tool used to bootstrap network update parameters
Learning rate                                  0.0001                           Tunes parameters in optimization algorithms
Loss function                                  Cross Entropy                    Evaluates the gap between the predicted value and the true value

Frontiers in Plant Science                                               06                                                                  frontiersin.org
Li et al.                                                                                           10.3389/fpls.2023.1256773

inference speed of the model. True positive (TP) means that the              3 Results and conclusions
predicted positive sample is actually a positive sample; false positive
(FP) indicates that the predicted positive sample is actually a              3.1 Results
negative sample; true negative (TN) means that the predicted
negative sample is actually a negative sample; and false negative                We selected several typically used CNN-based and ViT-based
(FN) means that the predicted negative sample is actually a positive         networks for comparison with our model. These include lightweight
sample.                                                                      networks such as SqueezeNet (Iandola et al., 2016), ShuffleNetV2
                                                                             (Ma et al., 2018), MobileNetV3 (Howard et al., 2019),
           Top-1 Accuracy = (TP + TN) / (TP + TN + FP + FN)     (4)         MobileFormer (Chen et al., 2022), EfficientNet (Tan and Le,
                                                                             2019), and Deit (Touvron et al., 2021) models. We also chose
                            Precision = TP / (TP + FP)           (5)         many heavyweight networks such as PoolFormer (Yu et al., 2022b),
                                                                             CVT (Wu et al., 2021), TNT (Han et al., 2021), and ResNet (He
                            Recall = TP / (TP + FN)              (6)         et al., 2016) for comparison. Additionally, we chose a wheat dataset
                                                                             with multiple components (such as roots, stems, and leaves) to
2.6 Experimental setup                                                       evaluate model performance on images depicting diverse
                                                                             conditions. The coffee dataset was employed to assess the
    All experiments run on a deep learning-based cloud platform.             performance of our method when confronted with complex
The hardware configuration is a 14-Core Intel(R) Xeon(R) Gold                backgrounds. Moreover, the rice dataset was used to investigate
6330 CPU @ 2.00 GHz, with 45 GB of RAM and an NVIDIA                         the classical binary classification problem.
GeForce RTX 3090 GPU. The operating system is Ubuntu 18.04,
and PyTorch 1.9.0 and Python 3.8 are used as software support.                   We chose the wheat dataset to verify the generalizability of
                                                                             PMVT under a real crop growth cycle. We can see from Table 3 that
                                                                             our proposed network achieved the best top-1 accuracy when
TABLE 3 Comparison of the PMVT model with other backbone models on three datasets (the FPS indicator is calculated on the desktop computer,
and bold text highlights the best-performing network).

Methods                                 Top-1 Accuracy(%)                    Parameters (M)   FLOPs (G) FPS (img/s)

SqueezeNet-1.0              Wheat              Coffee      Rice                         0.74
SqueezeNet-1.1                                                                          0.73
ShuffleNetV2-1.0              70.0               79.7       86.2                        1.27  0.73  293.0
MobileNetV3-Small                                           85.1                        1.54
PMVT-XXS (ours)               86.1               83.1       82.7                        0.98  0.26  311.5
ShuffleNetV2-1.5                                            89.7                        2.50
MobileFormer-26M              89.6               68.5       93.1                        2.22  0.15  151.9
MobileFormer-52M                                            86.2                        2.46
MobileFormer-96M              92.0               66.3       90.8                        3.33  0.06  170.2
MobileNetV3-Large                                           83.9                        4.22
EfficientNet-B0               93.6               85.4       87.3                        4.03  0.31  88.5
PMVT-XS (ours)                                              91.9                        2.01
ShuffleNetV2-2.0              92.5               73.0       88.5                        5.38  0.31  148.4
MobileFormer-151M                                           97.7                        6.34
EfficientNet-B1               91.4               77.5       91.4                        6.53  0.03  53.1
EfficientNet-B2                                             88.5                        7.72
Deit-Tiny                     92.8               79.2       90.8                        5.49  0.05  60.7
PoolFormer-S12                                              87.3                       11.39
                              92.8               84.2       84.0                              0.09  58.8
                                                            85.1
                              92.8               72.0                                         0.23  141.0

                              94.1               84.2                                         0.41  109.9

                              94.7               86.5                                         0.85  85.3

                              93.6               70.0                                         0.60  146.2

                              94.4               75.3                                         0.10  42.3

                              94.4               79.8                                         0.61  75.3

                              93.3               83.1                                         0.70  76.6

                              91.4               78.7                                         1.08  161.7

                              91.4               85.4                                         1.81  178.3

                                                                                                    (Continued)

Frontiers in Plant Science                                               07                         frontiersin.org
Li et al.                                                                                       10.3389/fpls.2023.1256773

TABLE 3 Continued                  Top-1 Accuracy(%)                     Parameters (M)   FLOPs (G) FPS (img/s)

 Methods                    Wheat  Coffee                 Rice                     19.63
                                                                                   23.40
  CVT-Tiny                  93.6                   82.0    86.2                    23.53  4.08  62.2
  TNT-Small                                                88.5                    42.50
  ResNet50                  92.8                   80.9    90.8                    5.06   4.85  67.3
  ResNet101                                                88.5
  PMVT-S (ours)             93.9                   70.8    92.0                           4.13  125.1

                            94.1                   63.0                                   7.86  66.3

                            94.9                   87.6                                   1.59  81.3

92.0%, whereas EfficientNet-B0 achieved a higher accuracy rate of        improved accuracy by 2.3% to reach 86.5%. Finally, the S version of
94.1%. Our PMVT reached state-of-the-art accuracy with rates of          the PMVT model achieved an accuracy rate of 87.6% on this
93.6 and 94.7, respectively. In comparing heavyweight networks, the      dataset; this was an improvement of 2.2% over that obtained by
PMVT model achieved an accuracy rate of 94.9% using only 5.06 M          the PoolFormer-S12 model. Figures 9 and 10 present the confusion
parameters, outperforming ResNet-101, which achieved an                  matrix, precision, and recall of the PMVT model. It can be seen
accuracy of 94.1% but used 42.5 M parameters. This proves that           from the figures that our model does not achieve satisfactory results
the proposed model is effective compared to the original                 in identifying red spider mite diseases.
MobileViT. Figure 6 presents the confusion matrix of our
proposed model. Figure 7 depicts the precision of the PMVT                   We applied the rice dataset to simultaneously testing the fitting
model, while Figure 8 illustrates its recall.                            ability of the PMVT model in a controlled laboratory environment
                                                                         and in a real natural condition. Surprisingly, the XS version of
    The coffee dataset was used to compare the performance of the        PMVT achieved 97.7% accuracy on this dataset, which was 5.8%
PMVT models in the field environment. As can be seen from                higher than the second-highest accuracy (obtained by the
Table 3, the traditional lightweight networks did not achieve            MobileNetV3-large model). In addition, the XXS version attained
acceptable accuracy rates. The XXS version of the PMVT model             an accuracy of 93.1%, which was 3.4% higher than the baseline of
achieved a top-1 accuracy rate of 85.4%, which was 3.5% higher           the MobileNetV3-small model. The S version of the PMVT model
than that of the SqueezeNet-1.1 model. Compared with the                 performed the worst, with an accuracy of 92%; however, it still
MobileFormer-96M model, the XS version of the PMVT model                 outperformed the ShuffleNetV2-2.0 model with similar parameters

FIGURE 6

Confusion matrix of the PMVT model on the wheat dataset.

FIGURE 7

Precision of the PMVT model on the wheat dataset.

Frontiers in Plant Science                                           08                         frontiersin.org
Li et al.                                                                 10.3389/fpls.2023.1256773

FIGURE 8

Recall of the PMVT model on the wheat dataset.

by 0.6%. Upon comparing models with similar sizes, we found that          significant amount of memory. Nevertheless, PMVT achieves the
the PMVT model has achieved the best accuracy rate. This proved           best accuracy with only 0.98M parameters. This makes it low-cost
that our model is very competitive on the classic binary                  and high-accuracy for plant disease identification. As artificial
classification problem. Figures 11 and 12 depict the confusion            intelligence technology advances, ViT can be better applied to the
matrix, precision, and recall of the PMVT model on the rice dataset.      visual task of plant disease identification.

    As seen in Table 3, our method does not excel in terms of FPS         3.2 Ablation studies
and FLOPs metrics. This because the self-attention mechanism
computes the weights between image patches, resulting in                      The data given in Table 4, it demonstrates the effectiveness of
numerous matrix calculations and multiplication operations                each module in our models. +Conv7x7 represents using a
during inference. Consequently, this increases the computational          convolution kernel of size 7 instead of the 3x3 convolution in
time. Additionally, because of the current immaturity of deep             the CNN block based on the MobileViT model. +CBAM uses
learning framework technology, numerous attention-weight
matrices must be stored and processed, thereby occupying a

FIGURE 9

Confusion matrix of the PMVT model on the coffee dataset.

FIGURE 10

Precision and recall of the PMVT model on the coffee dataset.

Frontiers in Plant Science                                            09  frontiersin.org
Li et al.                                                                                         10.3389/fpls.2023.1256773

      FIGURE 11

    Confusion matrix of the PMVT model on the rice dataset.

FIGURE 12

Precision and recall of the PMVT model on the rice dataset.

channel attention and spatial attention integrated in the ViT block      3.3 Conclusion
based on the MobileViT model. PMVT represents a new backbone
network built on the basis of MobileViT using both 7x7                       In this paper, we constructed a computationally efficient vision
convolution kernels and CBAM modules. It can be seen that each           transformer (ViT) model, referred to as PMVT, for the identification of
component can improve the accuracy of the model to                       plant diseases. Furthermore, larger convolution kernels and CBAM
varying degrees.                                                         modules enhanced the model's feature extraction capability.

TABLE 4 Ablation experiments investigating each component in the PMVT model (bold text highlights the best-performing network).

Methods                     Wheat(%)    Coffee(%)                        Rice(%)     Params(M)    FLOPs(G)

MobileViT-XXS               91.4        83.1                             92.0        0.96         0.27
+Conv7x7                    92.2(+0.8)  84.0(+1.1)                       92.8(+0.8)  0.97(+0.01)  0.30(+0.03)
+CBAM                       92.5(+1.1)  84.1(+1.0)                       92.6(+0.6)  0.97(+0.01)  0.27
PMVT-XXS                    93.6(+2.2)  85.3(+2.1)                       93.1(+1.1)  0.98(+0.02)  0.31(+0.04)
MobileViT-XS                93.3        84.2                             94.2        1.94         0.74
+Conv7x7                    93.9(+0.6)  85.3(+1.1)                       95.8(+1.6)  1.99(+0.05)  0.84(+0.1)
+CBAM                       93.6(+0.3)  85.6(+1.4)                       96.5(+2.3)  1.95(+0.01)  0.76(+0.02)
PMVT-XS                     94.7(+1.4)  86.5(+2.3)                       97.7(+3.5)  2.01(+0.07)  0.85(+0.11)
MobileViT-S                 93.9        84.3                             89.7        4.95         1.46
+Conv7x7                    94.4(+0.5)  85.4(+1.1)                       90.9(+1.2)  5.02(+0.07)  1.59(+0.13)
+CBAM                       94.4(+0.5)  84.7(+1.4)                       91.1(+1.4)  4.98(+0.03)  1.47(+0.01)
PMVT-S                      94.9(+1.0)  87.6(+3.3)                       92.0(+2.3)  5.06(+0.11)  1.59(+0.13)

Frontiers in Plant Science                                           10                                                          frontiersin.org
Li et al.                                                                                                 10.3389/fpls.2023.1256773

Comparative experiments were conducted on multiple datasets                                               research was funded by the major project of science and
containing images of plant diseases, thus demonstrating that PMVT                                         technology of Henan Province (Grant No. 221100110800), the
outperforms both lightweight and heavyweight networks. Additionally,                                      independent innovation project of Henan Academy of
PMVT outperforms both lightweight and heavyweight networks.                                               Agricultural Sciences (Grant No. 2023ZC067), and the innovation
PMVT has more powerful generalization capabilities and can be                                             team of Agricultural information technology (Grant
deployed on mobile devices for diagnosing plant diseases in field                                         No. 2023TD10).
environments. However, due to the shorter development time of
ViT, lightweight ViT models are comparatively slower than                                                 Acknowledgments
traditional lightweight CNNs when processing images. The
advancement of deep learning framework technology enables ViT to                                              We thank LetPub (www.letpub.com) for its linguistic assistance
perform computer vision tasks more effectively.                                                           during the preparation of this manuscript.

Data availability statement

    The original contributions presented in the study are included                                        Conflict of interest
in the article/supplementary material. Further inquiries can be
directed to the corresponding author.                                                                         The authors declare that the research was conducted in the
                                                                                                          absence of any commercial or financial relationships that could be
Author contributions                                                                                      construed as a potential conflict of interest.

    GL: Conceptualization, Methodology, Writing review &
editing. YW: Software, Visualization, Writing original draft. QZ:        Publisher's note
Data curation, Validation, Writing review & editing. PY:
Resources, Writing review & editing. BC: Funding acquisition,                All claims expressed in this article are solely those of the authors
Supervision, Writing review & editing.                                    and do not necessarily represent those of their affiliated
                                                                          organizations, or those of the publisher, the editors and the
Funding                                                                   reviewers. Any product that may be evaluated in this article, or
                                                                          claim that may be made by its manufacturer, is not guaranteed or
    The authors declare financial support was received for the            endorsed by the publisher.
research, authorship, and/or publication of this article. This

[References section omitted for brevity — full references present in extracted text above]

---

## EXTRACTED FIGURES (visual analysis)

### Figure 2 — Overview of the PMVT Model (page 4, `papers/06 - PMVT - CNN/inputs/figures/page4_fig2_blocks.png`)

**Caption (verbatim):** "Overview of the PMVT model. ↓2 means to downsample the feature map twice, and L stands for repeated stacking of L MobileViT blocks. For computer vision classification tasks, we use a classifier composed of an average pooling layer and a fully connected layer."

**Architecture flow transcribed from diagram (left to right):**

- **Input:** RGB Image (224×224×3)
- **Stem:** Conv2d_3×3 (labeled "Conv2d_3×3" — downsampling conv)
- **Layer 1:** Inverted Residual Block (single block, labeled "Inverted Residual Block")
- **Layer 2:** Three blocks in sequence: "Inverted Residual Block", "Inverted Residual Block", "Inverted Residual Block" — with notation L=2 below (appears to be 2 repeated InvRes blocks after the first downsample step)
- **Layer 3:** "Inverted Residual Block" followed by "Mobile ViT Block" — with notation L=4 below
- **Layer 4:** "Inverted Residual Block" followed by "Mobile ViT Block" — with notation L=4 below
- **Layer 5:** "Inverted Residual Block" followed by "Mobile ViT Block" — with notation L=3 below
- **Head:** Conv2d_1×1 (labeled "Conv2d_1×1") → Classifier (labeled "Classifier")

**Notes from diagram:** Downsampling (↓2) occurs at the start of Layers 2, 3, 4, and 5. The "L=" notation under each stage indicates the number of repeated MobileViT blocks stacked at that stage. Layer 2 has L=2, Layer 3 has L=4, Layer 4 has L=4 (note: from the diagram this reads L=4 for both layer 3 and layer 4), Layer 5 has L=3.

**Note:** Exact channel dimensions per layer are NOT labeled in the figure — they correspond to the MobileViT XXS/XS/S configurations from the original MobileViT paper (Mehta and Rastegari, 2021). The paper text only states the model follows MobileViT sizing conventions.

---

### Figure 3 — Structure of the Inverted Residual Block (page 5, `papers/06 - PMVT - CNN/inputs/figures/page5_fig3_full.png`)

**Caption (verbatim):** "Structure of the inverted residual block. C' represents the feature information obtained by convolving each channel of the feature map using a convolution kernel."

**Structure transcribed from diagram (left to right):**

- **Input:** Feature Map (multi-channel feature map shown as stacked planes)
- **Step 1:** Conv2d_1×1 (labeled "Conv2d_1X1" — expands channel dimension)
- **Step 2:** Depthwise Separable Convolution block (dashed border box labeled "Depthwise Separable Convolution") containing three parallel DWConv_7X7 blocks:
  - DWConv_7X7 → C' output
  - DWConv_7X7 → C' output
  - DWConv_7X7 → C' output
  (Three parallel DWConv_7×7 operations — these represent depthwise convolution applied to each channel group, with C' channels each)
- **Step 3:** Cat (concatenation of the three parallel branches)
- **Step 4:** Conv2d_1×1 (labeled "Conv2d_1X1" — restores channel dimension)
- **Output:** Feature map (same spatial resolution)

**Key observations:** The DWConv is 7×7 (not 3×3 as in original MobileViT). There are three DWConv_7×7 blocks shown in parallel within the depthwise separable conv box. The overall structure is: 1×1 Conv → 7×7 DW Conv (depthwise separable, shown as 3 parallel operations on channel groups) → Cat → 1×1 Conv.

---

### Figure 4A — Overall Structure of the MobileViT (Vision Transformer) Block (page 5, `papers/06 - PMVT - CNN/inputs/figures/page5_fig4A.png`)

**Caption (verbatim):** "The overall structure of the vision transformer block"

**Structure transcribed from diagram:**

- **Input:** Feature map [C, H, W] (labeled with C, H, W dimensions)
- **Local Representations branch (lower path):**
  - Conv2d_1×1 (labeled "Conv2d_1×1")
  - Conv2d_3×3 (labeled "Conv2d_3×3")
  - Fold → patch grid shown as colored tiles
  - Vision Transformer Encoder (×Lx — repeated L times)
  - Unfold → patch grid
  - Conv2d_1×1 (labeled "Conv2d_1×1") → output [C, H, W]
  - CBAM Block (labeled "CBAM Block", shown in blue)
- **Skip connection (upper path):** "Cat" — original feature map [C, H, W] concatenated with the transformer output → resulting in [2C, H, W]
- **After concatenation:** Conv2d_3×3 (labeled "Conv2d_3×3") → Output feature map [C, H, W]

**Flow summary:** Input → (split) → [lower: 1×1 Conv → 3×3 Conv → Fold → ViT Encoder×L → Unfold → 1×1 Conv → CBAM] + [upper: skip/Cat] → concat → 3×3 Conv → Output

---

### Figure 4B — Vision Transformer Encoder (page 5, `papers/06 - PMVT - CNN/inputs/figures/page5_fig4B.png`)

**Caption (verbatim):** "The structure of the vision transformer block encoder"

**Structure transcribed from diagram (Lx repeated block):**

- **Input:** Patch image (plant leaf image divided into patches shown)
- Linear (labeled "Liner") → Flatten → add Positional Encoding (⊕ with "~" symbol)
- **Repeated Lx block (dashed border):**
  - Layer Norm → Multi-head attention → residual add (⊕)
  - Layer Norm → MLP → residual add (⊕)
- **Output:** Feature representation

---

### Figure 4C — CBAM Block Architecture (page 5, `papers/06 - PMVT - CNN/inputs/figures/page5_fig4C_zoom.png`)

**Caption (verbatim):** "The architecture of the CBAM block, where ⊗ represents the multiplication with the original feature map."

**Structure transcribed from diagram:**

**Channel Attention (left box, labeled "Channel Attention"):**
- Input Feature Map [H, W] → Conv2d_3×3 (labeled "Conv2d_3X3") preprocessing step
- Parallel branches:
  - MaxPool → [C, H, W] → Shared MLP (labeled "Shared MLP") containing:
    - Conv2d_1×1 (labeled "Conv2d_1X1") — compress channels
    - Conv2d_1×1 (labeled "Conv2d_1X1") — expand channels back
  - AvgPool → [C, H, W] → same Shared MLP
- Output: C,R,H,W and C,H,W labels visible below the branches
- Cat (concatenate outputs of both pooling branches through MLP) → ⊕ → channel attention weights → ⊗ multiply with input feature map

**Spatial Attention (right box, labeled "Spatial Attention"):**
- Input from Channel Attention output
- Parallel branches:
  - MaxPool (labeled "MaxPool") → [1, H, W] feature map (blue block)
  - AvgPool (labeled "AvgPool") → [1, H, W] feature map (orange block)
- Cat (concatenate → [2, H, W])
- 7×7 Conv (from equation 3, not explicitly labeled in figure but consistent with text Eq.3)
- Sigmoid → ⊗ multiply with input
- Cat → Output Feature Map [H, W]

**Bottom:** "residual connection" label — indicating the final output includes a residual connection back to the original input Feature Map.
