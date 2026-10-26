# PAPER CONTENT: 20_TLMViT

**Source PDF:** papers/20_TLMViT/inputs/A_Deep_Features_Extraction_Model_Based_on_the_Transfer_Learning_Model_and_Vision_Transformer_TLMViT_for_Plant_Disease_Classification.pdf
**Extracted:** 2026-10-20

**Framework:** Keras / TensorFlow (Sec. VI: "Python 3.9.12 ... Spyder IDE including various libraries e.g., Keras, and TensorFlow"; CUDA 11.2, GTX 1080 Ti).
**Pretrained backbones (TLM stage, ImageNet weights, FROZEN, no FC head):** AlexNet (first 5 conv layers), ResNet50, VGG16, VGG19, InceptionV3 (five separate experiments).

---

Received 8 April 2023, accepted 24 April 2023, date of publication 5 May 2023, date of current version 12 May 2023.
Digital Object Identifier 10.1109/ACCESS.2023.3273317

A Deep Features Extraction Model Based on the
Transfer Learning Model and Vision Transformer
``TLMViT'' for Plant Disease Classification

AMER TABBAKH AND SOUBHAGYA SANKAR BARPANDA

School of Computer Science and Engineering, VIT-AP University, Amaravati 522237, India

Corresponding author: Soubhagya Sankar Barpanda (soubhagya.barpanda@vitap.ac.in)

ABSTRACT This paper proposes a novel approach for extracting deep features and classifying diseased
plant leaves. The agriculture industry is negatively impacted by plant diseases causing crop and economic
loss. Accurate and timely diagnosis is crucial for managing and controlling plant diseases, as traditional
methods can be costly and time-consuming. Deep learning-based tools effectively detect plant diseases
depending on the qualitative of extracted features. In this regard, a hybrid model for plant disease classi-
fication based on a Transfer Learning-based model followed by a vision transformer (TLMViT) is proposed.
TLMViT has four stages: 1) data acquisition, where the PlantVillage and wheat datasets are used to train
and evaluate the proposed model, 2) image augmentation to increase the number of training samples and
overcome the overfitting issue, 3) leaf feature extraction by two consecutive phases: initial features extraction
by using pre-trained based model and deep features extraction by using ViT model, and 4) classification
by using MLP classifier. TLMViT is experimented with using five pre-trained-based models followed by
ViT individually. TLMViT performs accurately in plant disease classification, obtaining 98.81% and 99.86%
validation accuracy for VGG19 followed by the ViT model on PlantVillage and wheat datasets respectively.
Moreover, TLMViT is compared with pre-trained-based architecture. The comparison result illustrates
that TLMViT achieved an enhancement of 1.11% and 1.099% in validation accuracy, 2.576% and 2.92%
in validation loss compared with the transfer learning-based model for PlantVillage and wheat datasets
respectively. Thereby proposed model proves the efficiency of using ViT for extracting deep features from
the leaf.

INDEX TERMS Plant disease, image processing, deep learning, transfer learning, vision transformer.

I. INTRODUCTION                                                         be the potential causes of plant diseases [4]. Phytopathology
Plant diseases have become more prevalent in recent years               is the scientific study of plant disease that focuses on ways to
due to globalization, trade, and climate change [1], [2]. These         treat and avoid the conditions responsible for plant illnesses.
issues have reached pandemic proportions in several nations,            To overcome plant illness, plants have to be diagnosed pre-
which increased the likelihood of crop damage and, in turn,             cisely. There are several methods available for diagnosing.
created a threat to people's access to adequate food and                Local plant clinics and agricultural groups have traditionally
nutrition [3]. Specialists should ensure to safeguard agri-             helped in disease detection. Still, the technique could be more
cultural plants. Parasitic organisms such as bacteria, fungi,           effective as there are more possibilities of human errors, and
viruses, roundworms (nematodes), and other plants can cause             it is difficult for humans to access plants across a wide area.
illnesses. Environmental conditions such as winter frosts or            Moreover, using software using machine learning techniques
summer dryness and lack or excess of nutrients in the soil can          can improve the efficiency of classifying diseased plants.

   The associate editor coordinating the review of this manuscript and     Smartphones are being developed using different tools
                                                                        and technology [6]. Modern plant illness classification
approving it for publication was Larbi Boubchir .                       approaches can be adopted using smartphones due to

VOLUME 11, 2023  This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License.  45377
                                 For more information, see https://creativecommons.org/licenses/by-nc-nd/4.0/
                                                                   A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

their incorporation of high computational techniques,              Section VII. The summary of this research is addressed in
high-resolution screens, and built-in accessories such as          section VIII.
HD cameras, which will be more effective for plant disease
detection. Plant diseases are effectively diagnosed using          II. RELATED WORK
machine learning techniques [4] as accurate diagnostic tests       Plant disease diagnostics using machine learning and deep
can be carried out, which benefits in preserving the resource.     learning have been a primary focus of recent research. The
Farmers can upload field images recognized by smartphones,         review papers [1], [2], [9], [10] provided a summary of dif-
and distinct software can be used for analyzing, diagnosing,       ferent novel approaches, developed existing algorithms used
and developing action plans.                                       to classify plant diseases, and sought to determine which clas-
                                                                   sification method would be the most useful for this challenge.
   Recently, image processing techniques with deep neural          Different datasets have been utilized to classify various plant
networks have been effectively used diversely and proven to        leaves. References [11], [12], and [13] proposed a model
be highly effective approaches in constantly monitoring the        to classify a tomato leaves disease. At the same time, Dey
health of plants and identifying signs of diseases in their early  et al. [14] concentrated on betel vines. References [15], [16],
stages [5]. A neural network considers an image of a diseased      [17], and [18] utilized Cassava disease. References [19], [20],
plant as an input and processes the image to produce a crop-       [21], [22], [23] focused on classifying potato leaf disease.
disease pair. Creating a deep network in such a manner that        References [24] and [25] performed the proposed work on
the network topology, functions (nodes), and edge weights          a dataset of Guava leaves. Singh and Misra [26] focused
accurately map the input to output is challenging. While           on some of the most common leaves of banana, jackfruit,
training deep neural networks, the network parameters are          lemon, mango, potatoes, tomato, and sapota. Sharif et al. [27]
adjusted in such a manner that the mapping is enhanced             experimented with the proposed work on the oranges infested
better over the training period. This complex computational        with scale, Plant Village Dataset, Citrus Diseases Image
process has recently seen several conceptual and practical         Gallery Dataset, and their collected citrus dataset. Hassan and
advancements that have dramatically increased its perfor-          Maji [28] utilized the plant village dataset (which includes
mance [7], [8]. One of these approaches is Vision Trans-           maize, potato, and tomato crops), the rice dataset, and the
former, which takes the whole image and extracts its features      cassava dataset were all used in their experimental work for
by splitting the image into multi patches, and from each patch,    the study. In contrast, The CropDP-181 dataset was devel-
the transformer encoder will extract the features. Extracting      oped by Kong et al. [29] by combining three existing datasets
the features from the whole image may take much more               totalling 123,987 images (from the AIChalle, Inaturalist, and
time and extract unnecessary features, e.g., a background of       IP102) that together contain 47 diseases and 134 pest cate-
the leaf. Above all, this paper presents a novel approach to       gories. In [30], the dataset was collected from various sources,
classify plant disease using pre-trained deep learning model       including (Apple plants, Wheat, Cotton, Maize, and Rice).
followed by ViT. The pre-trained model extracts the initial
features from the base convolutional networks that are already        Machine learning models are developed to classify
fine-tuned and reduces the dimensionality of the image mak-        plant leaf disease and achieve good performance. In this
ing the next stage less complex. Then the initialled features      regard, the following is a discussion of some methods.
are passed as input to ViT to extract the most significant fea-    Mokhtar et al. [11] proposed an approach that identifies
tures (deep features). A simple MLP is used to classify these      a diseased leaf without knowing the class of the dis-
deep features to which classes belong. The main contributions      ease. An essential feature of the tomato leaf is its sur-
of this paper are pointed out as follows:                          face texture, so GLCM is used where Energy, Contrast,
                                                                   Sum of Squares, Correlation, Entropy, Sum Entropy, Cluster
   1) Using a pre-trained model to reduce the dimensionality       Shade, Cluster Prominence, and Homogeneity are extracted.
       of the image, making the next stage less complex.           SVM is used as a classifier with different kernel functions.
                                                                   Bhargava et al. [31] applied different machine learning
   2) Extract the deep feature of the leaf using a combination     approaches, such as LR and SVM to classify varieties of
       of the pre-trained model and ViT making the classifier      leaves of vegetables and fruits. The dataset is segmented
       more accurate.                                              using grab-cut and fuzzy c-means clustering. Then the
                                                                   features were extracted by discrete wavelet transform, the
   3) Experiment with five different combinations of transfer      histogram of gradients, Laws' texture energy, textural, statis-
       learning-based models (such as ResNet50, AlexNet,           tical, and colorgeometrical with 114 features. PCA is used
       Inception V3, VGG16, and VGG19) with ViT.                   for feature selection to reduce the dimensionality of up to
                                                                   30 features. The classification results show the significance of
   4) Comparisons between the TLMViT and five other                feature selection and the increased classification recognition
       transfer learning-based models are made.                    rate. Sharif et al. [27] detected the lesion on leaves using
                                                                   an optimized weighted segmentation method on enhanced
   The rest of this paper is structured as follows. Section II     images obtained in preprocessing stage. Then the features
of the study offers a brief review of the relevant research.
Section IV introduces the background of the methods used.                                                                                                           VOLUME 11, 2023
Section V outlines the workflow and methodology used, fol-
lowed by the experimental parameters in Section VI. Exper-
iment outcomes with extensive discussion are presented in

45378
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

are extracted and fused using color (HSV, LAB, LUV,              algorithm based on multilevel thresholding with fuzzy, Tsal-
and HSI), geometric, and texture (18 GLCM features). The         lis, and Kapur's entropy for various fruit leaf segmentation.
best features are selected using principal component analysis    Khalifa et al. [37] augmented the dataset (from 1,722 to
(PCA) estimation. The multi-class support vector machine         9,822 images of potato leaves) using Reflection, Zoom, and
(M-SVM) is used to classify citrus diseases. They conclude       Gaussian noise functions. They proposed a simple CNN using
that enhancing the contrast of the lesion part will improve      two convolutional layers for feature extraction and two fully
segmentation accuracy. Bhargava and Bansal [32] focused          connected layers for classification achieving 98% accuracy
on segmenting the lesion from the mono-colored apples            as overall mean testing. Whereas, Rozaqi and Sunyoto [20]
using two algorithms, e.g., ``Otsu'' strategy and k- means       used four convolutions and one fully connected layer on
clustering. Then a combination of color coherence, Gabor         the same dataset of potato leaf. They achieved 97% and
wavelet, 14 geometric features, and 13 statistical & textural    92% accuracy on training and validation data, respectively.
features that are extracted from the segmented image are         Mittal and Gupta [38] focused on studies that show how
used as features. The proposed model is trained using linear     creating a complete and unusual image of a diseased leaf and
SVM. From the result, they conclude that the proposed model      increasing the dataset may help the classification network
achieves a good performance compared with other kernels of       perform better. A binary generator network is proposed to
the SVM classifier. In [21], the K-means algorithm extracts      address the issue of how a generative adversarial network
the infected regions from the leaf and then extracts the lesion  (GAN) generates a diseased leaf on a specific shape. Also,
features using GLCM. They considered feature extraction          using edge-smoothing and Image pyramid methods to over-
as a type of dimensionality reduction process. Pahurkar and      come the challenge of synthesising a complete lesion leaf
Deshmukh [30] propose a model that optimizes variance            image comprised of various synthetic edge pixels and net-
by fusing GLCM, edge map, color map, and convolutional           work out pixels. Khamparia et al. [19] benefited the properties
feature sets into an ensemble of features and then using         of the encoding part of the autoencoders model to extract
particle swarm optimization (PSO) to choose the optimal          useful features. They proposed a convolutional encoder net-
features. The parametric variant classification techniques are   work that only combines CNN with the encoding part. More-
also tuned with a Genetic algorithm (GA). Tabbakh and            over, 97.50% accuracy was achieved after 100 iterations.
Barpanda [33] proposed an approach of a machine-learning         Liang et al. [39] presented a network-based estimate method
model where wavelet transforms, GLCM methods, and sta-           for disease identification using ResNet50 and residual where
tistical features are used to extract different combinations     Shuffle-Net-network v2's architecture is used to reduce com-
of leaf features. Then the extracted features are utilized for   putational complexity. Johnson et al. [22] developed an auto-
training and comparing six machine-learning models, e.g.,        matic method for identifying blight disease patches on potato
SVM, AdaBoost, etc. They modified the GLCM approach to           leaves in field circumstances utilizing the Mask Region-based
focus on extracting the features of the leaf part only. SMOTE    convolutional neural network (Mask R-CNN) architecture
method is used to handle the imbalanced dataset. The highest     and a residual network as the backbone network. The Mask
result is achieved by using LGBM with 94.39% accuracy.           R-CNN model accurately distinguished between the infected
                                                                 area on the potato leaf and the similar-looking background
   Deep learning has proven to be an effective tool for          soil patches that often affect binary classification results. The
image classification tasks, outperforming traditional methods    proposed work is experimented after converting the original
such as support vector machines and random forests. Deep         dataset from RGB to YCrCb, XYZ, LAB, HSV, and HSL
learning algorithms require large datasets to train models       color spaces, giving a separate model for each color space.
effectively. In [34], cropping the lesion portion from the       Hassan and Maji [28] have suggested a new deep-learning
original image and making them a new sample is used to           model using the inception layer and residual connection. The
overcome the lack of dataset size. Whereas Geetharamani and      number of parameters in the proposed work is reduced by a
Pandian [35] augmented the dataset using various functions       margin of 70% using depthwise separable convolution, which
such as image flipping, scaling, rotation, principal compo-      directly impacts the computing cost.
nent analysis (PCA) color augmentation, noise injection, and
gamma correction. From the result, the authors observed that        Recent works [18], [40], [41], [42] applied the ViT concept
the data augmentation methods could improve the model            in agricultural applications. Reedha et al. [40] highlighted
performance. References [14], [26], and [36] used an image       how efficiently the convolutional-free ViT model, using the
segmentation algorithm to identify damaged leaf areas to         self-attention mechanism, interprets an image into a sequence
analyze better and have little processing effort to get optimum  of patches for processing by a standard transformer encode.
results. Dey et al. [14] found that using HVS color space        They obtained a high performance even though the dataset
on their dataset, the hue component shows clearly where          is small. They justified that due to data augmentation, trans-
a leaf has rotted. In addition, the hue component masks          fer learning, and a low number of classes. Wu et al. [41]
the background and the rest of the leaf region. Singh and        passed the dataset into two ViT models parallelly, where a
Misra [26] segmented the lesion by masking the mostly green      small patch size is used in one model, and a large patch
pixels. Chaturvedi et al. [36] proposed a modified firefly       size is used for another model. Fusion models combined

VOLUME 11, 2023                                                  45379
                                                                      A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

these two models to be fed into the MLP header. Conclud-                     disease images instead of applying the Vit model on
ing that, by combining different scales of the sequence of                   raw images with entire dimensions, which reduces the
self-attention, the model can extract more information from                  time consumption and trainable parameter numbers.
images from various granularities. Thai et al. [18] applied ViT          2) ViT model is used to extract the deep features of leaf
based on achieving a 90% F1-score. They used quantization                    images, i.e., interactions between all pixels that are
to make the model three times smaller before deploying it                    extracted from the previous stage (Initial Features).
on a Raspberry Pi 4 Model B. Yasamin et al. [42] combined a              3) Different combinations of Transfer learning-based
classical convolutional neural network (CNN) with ViT. They                  models with ViT have been experimented.
concluded that ViT gives good accuracy but decelerates the               4) Different five transfer learning-based models are com-
prediction, and this approach can recompense for the speed.                  pared with the proposed work TLMViT.

   Based on what is discussed in the related work, it can be          IV. PRELIMINARIES
noted that CNN models represent the process on images as              A. TRANSFER LEARNING
template matching, which extracts the neighbouring features           Transfer learning models or pre-trained models such as
and does not consider the relation between overall features.          ResNet50, AlexNet, Inception V3, VGG16, and VGG19 are
In contrast, ViT can extract and relate the features by giving        very active and useful in overcoming the flaws in deep
the position to patches of pixels and interactions between            learning [43]. Training a deep learning model with millions
them. ViT drawback is the need for a vast dataset and consum-         of parameters needs a lot of training samples that are only
ing much time due to its correlation between all pixels with          sometimes applicable to be collected and too much training
keeping the image scale without reduction. Moreover, pre-             time [44]. The transfer learning technique is utilized to trans-
trained models could be utilized significantly to extract the         fer the knowledge from a pre-trained model to another model,
features from the leaf and simultaneously reduce the dimen-           which should be trained for a classification task. Hence, the
sionality of the image, which will be considered as initial           training time is decreased because the model is not trained
features to be passed for the next stage to extract the deep          from scratch. Moreover, it is utilized to train the deep learning
features from them. Thus, combining both approaches could             model on a small-size dataset and overcome the overfitting
extract comprehensive and significant features to obtain an           issue [45], [46]. Fine-tunning is the most popular architecture
accurate model.                                                       in transfer learning, where a large dataset is used to pre-train
                                                                      the model and then, the parameters of the pre-trained model
III. MOTIVATION AND OBJECTIVES                                        are frozen and transferred to the target model for fine-tuning
Most of the DL-based research work in the plant disease               with the dense layers of the target model. Finally, the dense
classification field, which uses the convolution process, may         layers which have reasonable parameters should be trained on
be seen as template matching, whereby the same filtering              the dataset that belongs to the desired classification task [47].
template is applied to several parts of the same image. How-          Figure 1 shows the process of using the transfer learning
ever, the convolution layer represents only the connections           model for training a new model to classify a new dataset.
between neighbouring pixels since convolution is a local
action and is not able to encode the orientation and position         FIGURE 1. An illustration of using a Pre-Trained based model to classify a
of infected parts in the leaf. In case the new leaf has different     new dataset.
positions of infection from a trained set, then the classifier
would have a hard time classifying the diseased leaf.                 B. VISION TRANSFORMER (ViT)
                                                                      The transformer was initially used to visualize backbone
   In contrast, a transformer layer may overcome these issues         networks through the local relational network (LR-Net) [48]
by representing the interactions between all pixels, so the
transformer is considered a global operation. In a transformer,                                                                                                        VOLUME 11, 2023
the attention unit is an adaptive filter, and the filter weights are
set according to how well two pixels compose. The modelling
capabilities of this sort of computer module are superior.
Since Vision Transformer is applied on the whole image to
initialize the patches and process the attention patches, the
time-consuming of using Vision Transformer depends on the
size of the image and the number of patches. Hence, it is better
to reduce the dimension of an image, whereas scaling down
the dimension in an ordinary way will lead to the loss of some
significant features of the plant leaf in the image. The transfer
learning extracts the features and reduces the dimension of
the image before the classification part of its model. In light
of this, the aims listed below constitute the focus of the study:

   1) Different transfer learning models are used to extract
       initial features and reduce the dimensionality of plant

45380
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

and SASA [49]. Both models improved accuracy over ResNet           A. DATA ACQUISITION AND IMAGE AUGMENTATION
while utilizing the same theoretical computing resource by         In this work, PlantVillage and Wheat Rust datasets are used to
restricting self-attention computation to a local sliding win-     evaluate the proposed model for extracting the deep features
dow. LR-Net has the same theoretical and computational             of leaves.
complexity as ResNet, but it is much slower in reality. ViT
model was proposed by Dosovitskiy et al. [50] in 2021 as           1) PlantVillage
an efficient image classifier. He suggested using the origi-       The PlantVillage dataset is a collection of images of healthy
nal transformer architecture in computer vision applications       and diseased plants created by the PlantVillage non-profit
since it was previously applied only in natural language pro-      project. The dataset contains over 54,000 images of over
cessing applications. In image classification, ViT performs        38 different crop species, focusing on cassava, tomato, pep-
better than standard CNN architectures when trained on a           per, and potato. Each image is labelled with the species of
huge amount of image data. The input image is divided into         plant and the disease that is present if any. The dataset is
patches, each one is flattened and merged across the channels      freely available for computer vision and deep learning tasks
of the image to produce a vector representation of each patch.     such as Image classification, Object detection, and seman-
The patch embeddings are calculated via linear projecting          tic segmentation. In this research, three different crops of
the vectors using a dense layer. The positional embeddings         the PlantVillage dataset [51] from Kaggle are utilized and
are generated, which help the ViT model to analyze the             downloaded for training and evaluating the proposed model.
patches orderly and have a full view of the input image.           It comprises three main categories of plants: pepper bell,
Then, each patch is added to the corresponding positional          potato, and tomato leaves. It includes 20,638 images of dis-
embedding to obtain the input of the transformer's encoder.        eased and healthy leaves from these three crops. Table 1
The encoder consists of one block executed several times such      illustrates the differentiation of two bell pepper leaves, three
that the architecture of the block contains a multi-head self-     varieties of potato leaves, and ten types of tomato leaves.
attention followed by a dense layer. Finally, MLP classifies       Figure 3 shows samples of all 15 types of leaves included
the input image based on the output of the transformer's           in the dataset.
encoder.
                                                                      From Table 1, it can be noted that some classes have less
V. METHODOLOGY                                                     number of images compared to other classes. Moreover, the
This paper proposes a hybrid model for plant disease classi-       limitation of images while training the model leads to over-
fication based on transfer learning models and vision trans-       fitting issues. In order to restrain the influence of imbalanced
formers (TLMViT), as shown in Figure 2. TLMViT includes            data and overfitting issues, data augmentation is applied
four stages:                                                       to increase the samples [52]. The data augmentation, such
                                                                   as scaling, shearing, rotating, horizontal flip, zooming, and
   1) Data acquisition: The PlantVillage and Wheat Rust            shifting [53], [54] are applied, and its parameters are noted in
       datasets are used to train and validate the proposed        Table 5. From the initial set of 16511 training images, the data
       model. Different three crops are considered in the          augmentation generated 33257 augmented training images.
       PlantVillage dataset (pepper bell, potato, and tomato
       leaf).                                                      2) WHEAT RUST
                                                                   Wheat rust disease is a fungal disease that affects wheat
   2) Image augmentation: It artificially increases a dataset's    plants, causing significant damage to crops and resulting in
       size by applying random transformations to existing         significant economic losses for farmers. The disease is caused
       images. This allows the model to generalize better and      by several species of fungi belonging to the Puccinia genus.
       reduce overfitting, as it has seen different variations of  There are three main types of wheat rust: stem rust, leaf rust,
       the same image.                                             and stripe rust [55]. Stem rust is the most destructive type, as it
                                                                   attacks the stems of the wheat plant, weakening the plant and
   3) Feature extraction: It is a process of identifying and       reducing yield. Leaf rust affects the leaves of the plant and can
       extracting important information from raw data, like        reduce photosynthesis, while stripe rust affects the leaves and
       edges, corners, and textures in image data, which           stems of the plant. Wheat rust is spread by wind-borne spores,
       can be used as input for a classifier model. More-          which can travel long distances, making it difficult to control.
       over, two phases are consequently used to extract the       Preventing the spread of wheat rust is crucial to maintaining
       features:                                                   healthy crops and ensuring food security. Early detection
         a) First phase: a Pre-Trained based model is used to      and prompt treatment are essential for effectively managing
             extract the initial features of the leaf, then pass   the disease. In this work Wheat Rust dataset [42] is utilized
             them to the second stage.                             for training and evaluating the proposed model. The dataset
         b) Second phase: ViT is utilized to extract the deep      contains three classes (1128 Brown rust, 1348 Yellow rust,
             features from the initial features.                   and 1203 Healthy wheat), as illustrated in Table 2. Figure 4
                                                                   shows samples of the Wheat Rust dataset.
   4) Classification: simple MLP classifier is used to be
       trained and evaluated.                                                                                                                                     45381

   Each stage is discussed in detail as follows:

VOLUME 11, 2023
                                                                         A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

TABLE 1. Details of the leaves images used in the PlantVillage dataset.

FIGURE 2. The architecture of the proposed model TLMViT.

TABLE 2. Details of the leaves images used in the Wheat Rust dataset.    basic structure but have various layers. Resnet50 is shorthand
                                                                         for the version of Resnet that supports 50 neural network
B. FEATURES EXTRACTION                                                   layers. The dimension of the default input of ResNet50 is
The proposed model uses two consecutive phases for desir-                224*224*3. Five different stages based on convolution and
able feature extraction. In the first phase, the pre-trained             Residual Networks are applied, and after each stage, Max-
model is used for the initial extraction of leaf features and for        Pooling Layers are applied, which leads to reducing the
reducing the dimensionality of images, leading to decreased              dimension of the image by 2. The output of the final stage
time while applying the second phase. In the second phase,               of the default input gives a 7*7*2048 size. Whereas the
the ViT model is applied on initial features for deep extraction         dimension of the PlantVillage dataset is 256*256*3, the final
of features.                                                             stage of ResNet50 gives 8*8*2048. These are used as the
                                                                         initial feature and pass it as input to deep feature extraction
1) 1ST PHASE (INITIAL FEATURES EXTRACTION)                               using ViT.
In this phase, a pre-trained model is utilized to extract
initial features (IF) from the input images, where dif-                  b: AlexNet
ferent pre-trained models, namely (AlexNet, Res-Net 50,                  Alex Krizhevsky and his colleagues [53] published their find-
VGG-16, VGG-19, and Inception-V3) are experimented on                    ings in a research article titled Imagenet Classification with
the proposed work to observe which model will be superior.               Deep Convolution Neural Network, in which they suggested
The weights of all pre-trained models are initialized by using           the model. There are eight layers in the Alexnet model,
the ImageNet dataset and then used to extract the features               involving five layers based on convolution with a combina-
from the PlantVillage and Wheat Rust datasets as described               tion of max pooling, followed by three fully connected layers.
below:                                                                   Relu activation is used for each of these layers, excluding the
                                                                         output layer. In this work, only the first five layers are used
a: ResNet 50                                                             to extract the initial features from the PlantVillage dataset,
Residual Network is referred to as ResNet. Kaiming He,                   and the last three layers are eliminated. The dimension of
Xiangyu Zhang, Shaoqing Ren, and Jian Sun initially                      input images is 256*256*3, and the output of the last layers
described this novel neural network in [56]. There are several           is 8*8*256, as mentioned in Table 3.
distinct implementations of ResNet, all of which use the same
                                                                         c: VGG16 AND VGG19
45382                                                                    VGG contains a combination of convolutional and three fully
                                                                         connected layers. The name of the VGG model depends on
                                                                         how many layers are used in the model [57], e.g., 13 con-
                                                                         volutional and three fully connected layers produce VGG16,
                                                                         whereas 16 convolutional and three fully connected layers

                                                                                                                                                                          VOLUME 11, 2023
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

                                                                              TABLE 3. Details of the feature map size for default and our input size of
                                                                              the AlexNet model.

                                                                              TABLE 4. Details of the feature map size for default and our input size of
                                                                              the VGG16 and VGG19 models.

FIGURE 3. Image Samples of PlantVillage Dataset: (a) Tomato Spider            used for the classification part. Feng et al. [60] represent the
mites Two spotted spider mite, (b) Tomato Yellow Leaf Curl Virus,             structure of the Inception V3 model in detail, the default size
(c) Tomato Target Spot, (d) Tomato Sep-toria leaf spot, (e) Tomato            of an input image in the Inception V3 model is 299*299*3,
Bacterial spot, (f) Potato Late blight, (g) Tomato Late blight, (h) Tomato    and the final output size is 8*8*2,048.
Leaf Mold, (i) Tomato Early blight, (j) Tomato healthy, (k) Tomato mosaic
virus, (l) Potato healthy, (m) Potato Early blight, (n) Pepper bell healthy,     In contrast, the image size in the PlantVillage dataset is
and (o) Pepper bell Bacterial spot.                                           256*256*3, which is less than the default, leading to less
                                                                              output size of the feature map. Moreover, the output size
FIGURE 4. Image Samples of Wheat Rust Dataset: (a) Brown Rust,                of the features map for all previous pre-trained models that
(b) Yellow Rust, and (c) Healthy wheat.                                       were used in our experiments gives 8*8*c (c is 2048, 256,
                                                                              512, 512 for ResNet 50, AlexNet, VGG16, and VGG19 mod-
produce VGG19 [58]. A simple 3*3 convolution kernel is                        els, respectively) when using PlantVillage dataset. To main-
utilized across all layers to deepen the network and reduce                   tain consistency throughout all experiments, the images of
the number of parameters. A 224*244 RGB image is used as                      PlantVillage dataset are rescaled to the size of the default
VGG's default input. Moreover, the last three fully connected                 input of inception V3, which is 299*299*3, and fed as input
layers of both VGG16 and VGG19 models are not used in                         to extract the initial features in size of 8*8*2,048 to be used
our work. Table 4 shows the default input of both models                      in the second stage of the proposed model.
(VGG16 and VGG19), which is 224*224*3 and generates
7*7*512 as an output of the last layer, whereas the input size                   In contrast, the image dimensions of the Wheat Rust
of the PlantVillage dataset is 256*256*3, so the last layer                   dataset are in different scales, so it may be difficult to compare
gives 8*8*512 an output size as shown in Table 4. This output                 and analyze the data accurately. To address this issue, rescale
is considered the initial features of the models to be forwarded              function is used to ensure that all inputs (image dimensions)
to the next phase to extract the deep features.                               are of the same shape. The image dimensions of the Wheat
                                                                              Rust dataset are rescaled as the same image dimensions of
d: INCEPTION V3                                                               PlantVillage to have the same output dimensions in the 1st
In the image classification competition at the 2014 ILSVRC,                   phase of feature extraction with the size of 8*8*c.
Google presented a network named GoogLeNet that could
achieve the performance of human experts on the Ima-                          2) 2ND PHASE (DEEP FEATURES EXTRACTION)
geNet database [59]. An improved version of GoogLeNet got                     In this phase, the Vit model is applied on the initial features
inspired to generate the Inception-v3 model. It uses various                  that are extracted from the previous phase to extract deep
sizes of receptive kernels. By utilizing zero padding, the                    features of leaves images, where ViT is described as follows:
convolution operation's output size is maintained. Finally,                   As mentioned in the first phase, the features are extracted
filter concatenation produces the feature maps that will be                   by using transfer learning model, such that the extracted
                                                                              features IF are represented by a 2D image I  RLW C
VOLUME 11, 2023                                                               dimensions, where L is the length, W is the width of image
                                                                              and C is the number channels. The final dimensions of

                                                                                                                                                                             45383
                                                                          A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

the initial features that are extracted by pre-trained model              equations [5 - 6] and Figure 5.
e.g, AlexNet, VGG-16, VGG-19, Res-Net, and Inception-V3
are R88256, R88512, R88512, R882048, R662048 respec-                      Xi (j) = FH=4 (Norm (Zi)) + Zi;
tively. Then these features are fed as input to ViT model to be
processed according to the steps that are shown in Figure 2.              i = 0,    4 and j = 1,    8                (5)

   The first step is image partitioning, such that the extracted          Xi(j) = MLP (Norm (Xi (j))) + Xi (j) ;
features are split into non-overlapping patches. The initial
feature dimensions from the previous phase are 88c. Since                 i = 0,    4 and j = 1,    8                (6)
the dimension is not too big (8  8), and splitting it into lots
of patches will give less dimension with less information on              where FH=4 is the function of multi-head self-attention with
features, so the number of patches is chosen by taking the half           heads H = 4, MLP is multi-layer perceptron neural network
size of the I F with N (P) = 4. Each patch P has size 4  4  C             that uses two hidden layers with 128 neurons in the first layer
where C is the number of IF's channels and (4  4) are the                 and 64 neurons in the second layer. Norm is the normalization
dimensions of P. The process of image partitioning is shown               operation of vector. Xi(j) is the output of transformer's block
in equation [1].                                                          at the j-th iteration. At j=8 (the last iteration of encoder
                                                                          block), the deep features are obtained.
  IF = P1||P2||P3||P4; P  R44C and IF  R88C (1)

where C = 256, 512, 512, 2048, and2048 for AlexNet,
VGG-16, VGG-19, Res-Net, and inception respectively.
Then, every patch is flattened to a 1D vector such that, the
size of the vector is S = 4  4  C. Hence, four flattened
patches (vectors) are obtained with size S. Next, the patch
embeddings PEi are calculated by linear projecting on the
four vectors using dense layer that has 64 units as shown in
equation [2].

                                                                          FIGURE 5. Structure of the transformer encoder.

        PEi = Project(Pi) = Dense64 (Pi) ;                                C. CLASSIFICATION
                Pi  R4.4.C1, PEi  R64, i = 1,    4 (2)
                                                                          The classifier that has been used in the proposed model is
where i is the index of patch and the dimension of PEi is 64.             MLP head. Once the transformer encoder generates the deep
   The second step: in this step, the positional information              features in the previous stage, they are fed to the classifier,
                                                                          which determines to which class the image belongs. The deep
of patches Zi is retained by adding the patch embeddings PEi              feature vectors Xi(8) are fed to the MLP classifier as input
to the position embeddings PEi. The aim of this step is to                values, where the input layer has 256 neurons equal to the
process the patches as per their positions in the images (IF),            dimension of the deep feature vector Xi(8). Two hidden layers
and the process of getting positional information is shown in             with dimensions 1024 and 512 for the first and second layers
equation [3].                                                             respectively are used. Lastly, 15 neurons are used as the MLP
                                                                          classifier's output, which is the dataset's number of classes.
positional information(Zi) = PEi + Pos_Ei; i = 1,    4                 The process of classification is shown in equation [7].
                                                                     (3)

where PEi and Pos_Ei have the same dimension (64*1),                      P (j) = MLPhead Xi(8) ; j = 1,    15         (7)
and i is the index of patch. Position embeddings Pos_Ei are
generated by mapping integer numbers which represent the                  VI. EXPERIMENTAL ANALYSIS
                                                                          All the experiments in this work are carried out on Win-
positions of patches into vectors with size 64. Finally, the              dows 10 PCs version 21H2, with 64-bit operating system
                                                                          and a processor Intel(R) Xeon(R) 4.01 GHz with 64 GB
class token Z0 is embedded and appended to the positional                 RAM. CUDA with version 11.2 is used on NVIDIA Geforce
information Zi; i=1 . . . ,4 such as the dim ( Z0) = dim ( Zi);           GTX 1080 Ti. Python 3.9.12 was used as the programming
i = 1 . . . , 4 as shown in equation [4].                                 language and all the experiments were performed on the
                                                                          Spyder IDE including various libraries e.g., Keras, and Ten-
       Z = [Z0||Z1||Z2||Z3||Z4]  (4)                                      sorFlow. Table 5 illustrates the parameters of all models that
                                                                          were experimented in this research. In the data augmentation
   The third step is transformer encoding which consists of a             function, the parameters are randomly initialized in ranges.
one block that is executed N times (N = 8) to extract the deep            Dataset Splitted (80% for training and 20% for both eval-
features. This block has two main components i.e., multi-                 uating and testing the model). Since the dataset size is not
head self-attention and MLP beside that, normalization oper-
ations are used to improve the performance of transformer
encoder. The process of transformer encoder is shown in

45384                                                                                                                      VOLUME 11, 2023
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

small, 20% would be sufficient for testing the model (almost      PlantVillage dataset. Figure 6 (a, c, e, g, and i) show the
4100 images for testing the model). The Epoch number is           classification accuracy of training and validation set for five
chosen (25) to compare the results of all experiments at a        experiments, i.e., ResNet50, AlexNet, VGG16, VGG19, and
specific epoch number to give a fair comparison. Since the        Inception V3, each of which is followed by ViT-based model
dataset is under multi-class classification problems, categori-   respectively in 25 epochs. The outcome of training accuracy
cal cross-entropy is used as a loss function that calculates the  is improved after each epoch till it reaches 90.405%, 92.5%,
predicted probability distribution and the true probability dis-  99.60%, 98.8%, and 99.2% for ResNet50, AlexNet, VGG16,
tribution. Adam is used as an optimizer since it requires fewer   VGG19, and Inception V3, each of which is followed by
parameters for tuning, which leads to faster computation time.    ViT-based model respectively. The outcome of validation
The dimension of the initial feature (features dimension at the   accuracy is also improved after each epoch till it reaches
last stage of the pre-trained model) is 8*8*C, as explained in    89.2%, 90.49%, 98.43%, 98.81%, and 98.48% for ResNet50,
section V-B1. So, the patch size of the ViT model is chosen       AlexNet, VGG16, VGG19, and Inception V3, each of which
four as half of the initial feature dimension. The rest of the    is followed by ViT-based model respectively as shown in
parameters are chosen experimentally.                             Table 6. Figure 6(b, d, f, h, and j) show the training and
                                                                  validation loss outcome in 25 epochs for five experiments,
TABLE 5. Experimental parameters.                                 i.e., ResNet50, AlexNet, VGG16, VGG19, and Inception V3,
                                                                  each of which is followed by ViT-based model respectively.
VII. RESULTS AND DISCUSSION                                       The outcome of training loss is decreased after each epoch
This section discusses the results of the proposed model and      till it reaches 0.3448, 0.2288, 0.0814, 0.1, and 0.0900 for
compares its performance with the pre-trained-based models'       ResNet50, AlexNet, VGG16, VGG19, and Inception V3,
performance.                                                      each of which is followed by ViT-based model respectively.
                                                                  The outcome of validation loss is decreased after each epoch
A. RESULTS OF OUR PROPOSED MODEL                                  till it reaches 0.3735, 0.2974, 0.1300, 0.0947, and 0.1 for
The proposed model based on a pre-trained model followed          ResNet50, AlexNet, VGG16, VGG19, and Inception V3,
by ViT is evaluated in terms of accuracy, loss, precision, F1-    each of which is followed by ViT-based model respectively
score, and recall. Five different pre-trained-based models are    as shown in Table 6. Moreover, according to the results
experimented to measure the performance of the proposed           in the five experiments, (VGG-16 followed by ViT) model
model. Firstly, the weights of the CNN model are initialized      outperforms in the training phase where the training accuracy
by the frozen weights of a transfer learning-based model that     is equal to (99.60%), and training loss is equal to (0.0814).
was trained on the ImageNet dataset. Hence, no training is        Whereas, (VGG-19 followed by ViT) model outperforms in
required in the initial features extraction phase, this manner    the validation phase where the validation accuracy is equal to
overcomes the issue of consuming a lot of processing time for     (98.81%), and validation loss is equal to (0.0947).
training a complex model from scratch on the PlantVillage
dataset, which contains more than 20 thousand images. The            Furthermore, Figure 8 shows the confusion matrices of the
aim of using pre-trained-based models in our approach is          proposed model that experimented using ResNet50, AlexNet,
to (1) Extract the initial features from the input image and      VGG16, VGG19, and Inception V3 each of which is followed
reduce the number of trainable parameters. (2) Reduce the         by ViT-based model respectively. Finally, some statistical
dimension of the input image to accelerate the performance        manners are used to analyze confusion matrices such as (pre-
of ViT in the next phase and drive Vit to focus on the most       cision, recall, and F1-score) for ResNet50, AlexNet, VGG16,
significant features. Then, ViT model is trained to extract       VGG19, and Inception V3 each of which is followed by
the deep features and classify the plant diseases based on        ViT-based models respectively as shown in Table 8.
the training set of the PlantVillage dataset. After that, the
overall model is validated based on the testing set of the        B. RESULTS OF TRANSFER LEARNING-BASED MODELS
                                                                  In this work, five different pre-Trained based models are
VOLUME 11, 2023                                                   also experimented to classify the plant diseases on the
                                                                  PlantVillage dataset, which means the extracted features
                                                                  that used in its model are the initial features only as men-
                                                                  tioned previously. The performance of five different trans-
                                                                  fer learning-based models i.e, ResNet50, AlexNet, VGG16,
                                                                  VGG19, and Inception V3 are shown in Table 6. The clas-
                                                                  sification accuracy of the training set is increased after
                                                                  each epoch till it reaches 80.43%, 83.9%, 90.44%, 88.70%,
                                                                  and 92.78% for ResNet50, AlexNet, VGG16, VGG19, and
                                                                  Inception V3 respectively, and the classification accuracy
                                                                  of validation set is also improved after each epoch till it
                                                                  reaches to 79.84%, 81.91%, 89.31%, 86.55%, and 90.77%
                                                                  for ResNet50, AlexNet, VGG16, VGG19, and Inception V3

                                                                                                                                                                 45385
       A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

FIGURE 6. The performance of TLMViT models. (a, c, e, g, and i) Accuracy and (b, d, f, h, and j) Loss of ResNet50_ViT, AlexNet_ViT, VGG16_ViT, VGG19_ViT,
and Inception V3_ViT models respectively.
TABLE 6. Performance comparison for training and validation set on PlantVillage dataset in terms of accuracy and loss between transfer learning-based
models and TLMViT.

TABLE 7. Performance comparison for training and validation set on Wheat Rust dataset in terms of accuracy and loss between transfer learning-based
models and TLMViT.

45386  VOLUME 11, 2023
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

FIGURE 7. The performance of Transfer Learning-based models. (a, c, e, g, and i) Accuracy and (b, d, f, h, and j) loss of ResNet50, AlexNet, VGG16, VGG19,
and Inception V3 models respectively.

TABLE 8. Results Comparison in terms of F1-score, precision, and recall  TABLE 9. Results Comparison in terms of F1-score, precision, and recall
between transfer learning-based models and TLMViT using PlantVillage     between transfer learning-based models and TLMViT using Wheat Rust
dataset.                                                                 dataset.

respectively. Figure 7( a, c, e, g, and i) show the classifi-            and Inception V3 respectively. The value of validation loss
cation accuracy of training and validation set in 25 epochs              is decreased after each epoch till reaches 0.6, 0.55, 0.3188,
for ResNet50, AlexNet, Inception V3, VGG16, and VGG19                    0.3792, and 0.2967 for ResNet50, AlexNet, VGG16, VGG19,
respectively. Whereas Figure 7( b, d, f, h, and j) show clas-            and Inception V3 respectively as shown in Table 6. Moreover,
sification loss of training and validation set in 25 epochs              the values of precision, F1-score, and recall for ResNet50,
for ResNet50, AlexNet, Inception V3, VGG16, and VGG19                    AlexNet, Inception V3, VGG16, and VGG19 are shown in
based models respectively. The value of training loss is                 Table 8. Moreover, a different comparison is done for the
decreased after each epoch till reaches 0.5492, 0.47, 0.2605,            pre-trained based models with each other, on the PlantVillage
0.3277, and 0.2064 for ResNet50, AlexNet, VGG16, VGG19,                  dataset. According to the results, Inception V3 achieves the

VOLUME 11, 2023                                                                                                                                                         45387
       A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

       FIGURE 8. The confusion matrix of the proposed model. (a) ResNet50 with ViT, (b) AlexNet with ViT, (c) VGG16 with ViT,
       (d) VGG19 with ViT, (e) Inception V3 with ViT.

45388                                                                                                                          VOLUME 11, 2023
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model
TABLE 10. A comparison between the proposed model vs. other works.

best performance among all, obtaining the highest training             Similarly, the proposed approach is experimented on the
accuracy equal to 92.78%, and the lowest training loss equal        Wheat Rust dataset. Table 7 presents the performance com-
to 0.2064 values, as well as the highest validation accuracy        parison for the training and validation set of the Wheat
equal to 98.48% and lowest validation loss equal to 0.1.            Rust dataset in terms of accuracy and loss between transfer
                                                                    learning-based models and TLMViT. Where the outcomes
   Furthermore, our proposed model is compared with the             of training accuracy are equal to 93.26%, 95.48%, 99,93%,
pre-trained based models. According to the results, the per-        99.67%, and 99.81% for ResNet50, AlexNet, VGG16,
formance of the proposed model outperforms the same model           VGG19, and Inception V3 each of which is followed by
without using ViT, due to extracting the deep features from         ViT-based model respectively. Furthermore, the outcomes of
the initial features by ViT. The initial features are extracted     validation accuracy are equal to 91.74%, 93.39%, 99.44%,
from the previous phase by the Transfer Learning-based              99.86%, and 99.69% for ResNet50, AlexNet, VGG16,
model. In other words, ViT in the proposed model's topology         VGG19, and Inception V3 each of which is followed by
is considered the second level of extracting features of leaf       ViT-based model respectively. Moreover, Table 9 shows the
images. By analyzing the comparison above, the proposed             results comparison between transfer learning-based models
model enhances performance compared with the Transfer               and TLMViT on the Wheat Rust dataset in terms of F1-
learning-based model in terms of accuracy and loss for plant        score, precision, and recall. Table 12 compares the epochs
disease classification. Where enhancements percentages are          number and execution time of a model taken to reach a
(1.11%, 1.10%, 1.08%, 1.10%, and 1.14% validation accu-             specific accuracy for both the pre-trained-based models and
racy) and (1.60%, 1.89%, 2.9%, 2.45%, and 4.00% valida-             the proposed model on the Wheat Rust dataset. The specific
tion loss) for ResNet50, AlexNet, Inception V3, VGG16,              accuracy is chosen as per the highest accuracy of the pre-
and VGG19 respectively. On average, the TLMViT model                trained-based model. For example, ResNet50 took 21 epochs
has 1.106% higher validation accuracy and 2.568% lower              to achieve 84% accuracy in 13.15 minutes, whereas the
validation loss than pre-trained-based models.                      proposed model using ResNet50 with ViT reached almost the
                                                                    same accuracy in 2 epochs in 1.39 minutes. According to the
   Table 11 compares the epochs number and execution time           results of all models that are experimented on the PlantVil-
of a model taken to reach a specific accuracy for both the          lage and Wheat Rust datasets, the proposed model performs
pre-trained-based models and the proposed model on the              accurately on classification problems for both datasets using
PlantVillage dataset. The specific accuracy is chosen as per        less number of Epochs and execution time.
the highest accuracy of the pre-trained-based model. For
example, ResNet50 took 15 epochs to achieve 80% accu-                  In addition, Table 10 compares the performance of our
racy in 56.90 minutes, whereas the proposed model using             model with other models using the main performance met-
ResNet50 with ViT reached almost the same accuracy in               rics (accuracy, precision, recall, and F1-score). In this
2 epochs in 7.79 minutes.

VOLUME 11, 2023                                                     45389
                                                                   A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

TABLE 11. Comparison of Epochs number and execution time for both  plantVillage and Wheat datasets respectively, which outper-
Transfer Learning-based model and TLMViT model using PlantVillage  formed the previous models.
dataset.

TABLE 12. Comparison of Epochs number and execution time for both  VIII. CONCLUSION
Transfer Learning-based model and TLMViT model using Wheat Rust    This paper proposes a hybrid model (TLMViT) for plant
dataset.                                                           disease classification based on a pre-trained based model
                                                                   followed by a vision transformer. TLMViT has four stages:
comparison, the ImageNet dataset is used to fine-tune the          data acquisition, image augmentation, feature extraction, and
parameters for our proposed model and all comparative works        classification. Firstly, three crops namely( bell peppers, pota-
that used the transfer learning approaches. As shown in            toes, and tomato leaves) including fifteen classes of the
Table 10, the recent works that are done so far on plant dis-      PlantVillage dataset and three classes of the Wheat Rust
ease classification concentrate on using four different scopes:    dataset, are used individually to train and evaluate our pro-
machine learning models, CNN models, CNN along with                posed model. Secondly, the leaf images in the datasets are
attention approach, and CNN along with ViT. The plant              augmented by many functions such as rotation, shifting,
disease-based machine learning classifiers need more steps         shearing, zooming, and flipping to increase the number of
in pre-processing stage to extract the leaf features, and the      training samples. Thirdly, the features are extracted in two
model performance would be less accurate due to the limita-        consecutive phases: initial features extraction and deep fea-
tion of machine learning. In [33], a machine learning model        tures extraction. In the first phase, the pre-trained model is
using LGBM mode is proposed and trained on the PlantVil-           used to extract the features of the leaf and call them initial
lage dataset, the performance metrics did not exceed 94%.          features. The weights of the pre-trained model are fine-tuned
Though CNN models are effective in classifying plant disease       using the ImageNet dataset. In the second phase of features
images, these models represent only the connections between        extraction, the vision transformer is utilized as deep layers to
neighbouring pixels and are not able to encode the orientation     extract the deep features of leaves based on initial features.
and position of infected parts in the leaf. For this reason,       Lastly, the MLP head classifier determines to which class the
their performance, i.e., accuracy, precision, recall, and F1-      leaf belongs. The proposed model's performance is exper-
score are not very close to 1. In [61], [62], [63], and [64],      imented by five pre-trained models and evaluated in terms
CNN models are trained on the PlantVillage dataset, where          of accuracy, loss, precision, F1-score, and recall. The results
the performance metrics did not reach 99%. Similarly, in [68],     show that the proposed model performs accurately in plant
[69], [70], and [71] different types of CNNs are trained on the    disease classification, getting the highest validation accuracy
Wheat dataset where the performance metrics did not reach          for VGG-19, followed by the ViT model. The validation
99%. Combining the attention approach along with the CNN           accuracy is equal to 98.81% and 99.86%, and validation
or pre-trained CNN model improves the accuracy because             loss is equal to 0.0947 and 0.0715 for PlantVillage and
the attention unit is considered an adaptive filter, where its     wheat datasets respectively. Moreover, the results of TLMViT
weights are set according to how well two pixels compose.          are compared with transfer Learning based models (without
In [65] and [66] DenseNet and pre-trained MobileNet-V2             using ViT) to show the efficiency of using ViT for deep fea-
along with attention mechanism are used respectively. In their     ture extraction. The TLMViT model outperforms the Transfer
proposed work, the performance metrics got improved and            Learning-based model with an enhancement of 1.11% and
reached 98% and 97% approximately. Finally, the pre-trained        1.099% higher in validation accuracy and 2.576% and 2.92%
CNN model along with ViT achieves the best performance             lower in validation loss for PlantVillage and wheat datasets
in plant disease classification where the metrics are close        respectively. From all the above, the following findings of
to 1. In [67], a CNN model followed by ViT is trained on           the proposed model can be noted: (1) The data augmentation
PlantVillage for plant disease detection, and the performance      approach provides a simple way to overcome the lack of
metrics reached 99% approximately. Whereas our proposed            images that leads to overfitting issues and reduce the influ-
model uses pre-trained models followed by ViT is trained           ence of an imbalanced dataset. (2) A pre-trained-based model
on PlantVillage and Wheat datasets. The performance met-           could be utilized to extract the initial leaf features and reduce
rics achieved accuracy equal to 98.81% and 99.86% for              the dimensionality of the original image, then be used as
                                                                   inputs for deep layers or a deep approach to extract the deep
45390                                                              features. (3) ViT model shows the ability to extract the deep
                                                                   features from extracted features of the CNN model (initial
                                                                   features). On the other hand, the pre-trained model often fine-
                                                                   tunes for specific tasks. So, the proposed model could need
                                                                   more training and fine-tuning the weights in the first phase of
                                                                   feature extraction for particular crops. In future work, various
                                                                   crops could be considered for training on TLMViT to study
                                                                   the ability of the proposed model.

                                                                                                                                                                    VOLUME 11, 2023
A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

ACKNOWLEDGMENT                                                                        [20] A. J. Rozaqi and A. Sunyoto, ``Identification of disease in potato leaves
The authors would like to thank the High-Performance Com-                                    using convolutional neural network (CNN) algorithm,'' in Proc. 3rd Int.
puting Laboratory, VIT-AP University, for providing the                                      Conf. Inf. Commun. Technol. (ICOIACT), Nov. 2020, pp. 7276.
resources and support that enabled this research to be con-
ducted. The availability of advanced computing tools and                              [21] A. Singh and H. Kaur, ``Potato plant leaves disease detection and classi-
skilled technical assistance played a crucial role in the suc-                               fication using machine learning methodologies,'' in Proc. IOP Conf. Ser.,
cessful completion of this article.                                                          Mater. Sci. Eng., vol. 1022, 2021, Art. no. 012121.

REFERENCES                                                                            [22] J. Johnson, G. Sharma, S. Srinivasan, S. K. Masakapalli, S. Sharma,
                                                                                             J. Sharma, and V. K. Dua, ``Enhanced field-based detection of potato blight
 [1] A. Kamilaris and F. X. Prenafeta-Bold, ``Deep learning in agriculture:                 in complex backgrounds using deep learning,'' Plant Phenomics, vol. 2021,
       A survey,'' Comput. Electron. Agricult., vol. 147, pp. 7090, Apr. 2018,              pp. 113, Jan. 2021.
       doi: 10.1016/j.compag.2018.02.016.
                                                                                      [23] T.-Y. Lee, I.-A. Lin, J.-Y. Yu, J.-M. Yang, and Y.-C. Chang, ``High effi-
 [2] Z. Iqbal, M. A. Khan, M. Sharif, J. H. Shah, M. H. Ur Rehman,                           ciency disease detection for potato leaf with convolutional neural net-
       and K. Javed, ``An automated detection and classification of cit-                     work,'' Social Netw. Comput. Sci., vol. 2, no. 4, pp. 111, Jul. 2021.
       rus plant diseases using image processing techniques: A review,''
       Comput. Electron. Agricult., vol. 153, pp. 1232, Oct. 2018, doi:              [24] A. Alharbi, ``AI-driven framework for recognition of guava plant diseases
       10.1016/j.compag.2018.07.032.                                                         through machine learning from DSLR camera sensor based high resolution
                                                                                             imagery,'' Sensors, vol. 21, no. 11, p. 3830, 2021.
 [3] S. P. Mohanty, D. P. Hughes, and M. Salath, ``Using deep learning for
       image-based plant disease detection,'' Frontiers Plant Sci., vol. 7, p. 1419,  [25] B. Srinivas, ``Prediction of guava plant diseases using deep learning,''
       Sep. 2016, doi: 10.3389/fpls.2016.01419.                                              in Proc. 3rd Int. Conf. Commun. Cyber Phys. Eng. Cham, Switzerland:
                                                                                             Springer, 2021, pp. 14951505.
 [4] J. G. A. Barbedo, ``A review on the main challenges in automatic plant dis-
       ease identification based on visible range images,'' Biosyst. Eng., vol. 144,  [26] V. Singh and A. K. Misra, ``Detection of plant leaf diseases using image
       pp. 5260, Apr. 2016, doi: 10.1016/j.biosystemseng.2016.01.017.                       segmentation and soft computing techniques,'' Inf. Process. Agricult.,
                                                                                             vol. 4, pp. 4149, Mar. 2017.
 [5] A. K. Pradhan, S. Swain, and J. K. Rout, ``Role of machine learning and
       cloud-driven platform in IoT-based smart farming,'' in Machine Learning        [27] M. Sharif, M. A. Khan, Z. Iqbal, M. F. Azam, M. I. U. Lali, and
       and Internet of Things for Societal Issues. Berlin, Germany: Springer,                M. Y. Javed, ``Detection and classification of citrus diseases in agriculture
       2022, pp. 4354.                                                                      based on optimized weighted segmentation and feature selection,'' Com-
                                                                                             put. Electron. Agricult., vol. 150, pp. 220234, Jul. 2018.
 [6] G. C. Index and C. Profiles, ``Geneva: International telecommunication
       union,'' Tech. Rep., Dec. 2015.                                                [28] S. M. Hassan and A. K. Maji, ``Plant disease identification using a novel
                                                                                             convolutional neural network,'' IEEE Access, vol. 10, pp. 53905401,
 [7] Y. Bengio and Y. LeCun, ``Scaling learning algorithms towards AI,''                     2022.
       Large-Scale Kernel Mach., vol. 34, no. 5, pp. 141, 2007, doi:
       10.1038/nature14539.                                                           [29] J. Kong, H. Wang, C. Yang, X. Jin, M. Zuo, and X. Zhang, ``A spatial
                                                                                             feature-enhanced attention neural network with high-order pooling rep-
 [8] J. Schmidhuber, ``Deep learning in neural networks: An overview,'' Neural               resentation for application in pest and disease recognition,'' Agriculture,
       Netw., vol. 61, pp. 85117, Oct. 2015, doi: 10.1016/j.neunet.2014.09.003.             vol. 12, no. 4, p. 500, Mar. 2022.

 [9] A. Bhargava and A. Bansal, ``Fruits and vegetables quality evaluation using      [30] A. Pahurkar and R. Deshmukh, ``PDMBM: Design of a high-efficiency
       computer vision: A review,'' J. King Saud Univ.-Comput. Inf. Sci., vol. 33,           plant disease classification method using multiparametric bio inspired
       no. 3, pp. 243257, Mar. 2021.                                                        modelling,'' in Proc. Int. Conf. Sustain. Comput. Data Commun. Syst.
                                                                                             (ICSCDS), Apr. 2022, pp. 16071615.
[10] A. Ahmad, D. Saraswat, and A. El Gamal, ``A survey on using deep
       learning techniques for plant disease diagnosis and recommendations            [31] A. Bhargava, A. Bansal, and V. Goyal, ``Machine learningbased detection
       for development of appropriate tools,'' Smart Agricult. Technol., vol. 3,             and sorting of multiple vegetables and fruits,'' Food Anal. Methods, vol. 15,
       Feb. 2023, Art. no. 100083.                                                           no. 1, pp. 228242, Jan. 2022.

[11] U. Mokhtar, ``SVM-based detection of tomato leaves diseases,'' in Intel-         [32] A. Bhargava and A. Bansal, ``Machine learning based quality evaluation
       ligent Systems (Advances in Intelligent Systems and Computing). Cham,                 of mono-colored apples,'' Multimedia Tools Appl., vol. 79, nos. 3132,
       Switzerland: Springer, 2015, pp. 641652.                                             pp. 2298923006, Aug. 2020.

[12] R. Karthik, M. Hariharan, S. Anand, P. Mathikshara, A. Johnson, and              [33] A. Tabbakh and S. S. Barpanda, ``Evaluation of machine learning models
       R. Menaka, ``Attention embedded residual CNN for disease detection in                 for plant disease classification using modified GLCM and wavelet based
       tomato leaves,'' Appl. Soft Comput., vol. 86, Jan. 2020, Art. no. 105933.             statistical features,'' Traitement Du Signal, vol. 39, no. 6, pp. 18931905,
                                                                                             Dec. 2022, doi: 10.18280/ts.390602.
[13] T.-T. Tran, J.-W. Choi, T.-T. Le, and J.-W. Kim, ``A comparative study of
       deep CNN in forecasting and classifying the macronutrient deficiencies on      [34] J. G. A. Barbedo, ``Factors influencing the use of deep learning
       development of tomato plant,'' Appl. Sci., vol. 9, no. 8, p. 1601, Apr. 2019.         for plant disease recognition,'' Biosyst. Eng., vol. 172, pp. 8491,
                                                                                             Aug. 2018.
[14] A. K. Dey, M. Sharma, and M. R. Meshram, ``Image processing based leaf
       rot disease, detection of betel vine (Piper BetleL.),'' Proc. Comput. Sci.,    [35] G. Geetharamani and A. Pandian, ``Identification of plant leaf diseases
       vol. 85, pp. 748754, Jan. 2016.                                                      using a nine-layer deep convolutional neural network,'' Comput. Electr.
                                                                                             Eng., vol. 76, pp. 323338, Jun. 2019.
[15] A. Ramcharan, K. Baranowski, P. McCloskey, B. Ahmed, J. Legg, and
       D. P. Hughes, ``Deep learning for image-based cassava disease detection,''     [36] R. Chaturvedi, A. Sharma, A. Bhargava, J. Rajpurohit, and P. Gothwal,
       Frontiers Plant Sci., vol. 8, p. 1852, Oct. 2017.                                     ``Multi-level segmentation of fruits using modified firefly algorithm,''
                                                                                             Food Anal. Methods, vol. 56, pp. 28912900, Nov. 2022.
[16] A. Ramcharan, P. McCloskey, K. Baranowski, N. Mbilinyi, L. Mrisho,
       M. Ndalahwa, J. Legg, and D. P. Hughes, ``A mobile-based deep learning         [37] N. E. M. Khalifa, M. H. N. Taha, L. M. Abou El-Maged, and
       model for cassava disease diagnosis,'' Frontiers Plant Sci., vol. 10, p. 272,         A. E. Hassanien, ``Artificial intelligence in potato leaf disease classifi-
       Mar. 2019.                                                                            cation: A deep learning approach,'' in Machine Learning and Big Data
                                                                                             Analytics Paradigms: Analysis, Applications and Challenges. Berlin, Ger-
[17] O. O. Abayomi-Alli, R. Damasevicius, S. Misra, and R. Maskeliunas,                      many: Springer, 2021, pp. 6379.
       ``Cassava disease recognition from low-quality images using enhanced
       data augmentation model and deep learning,'' Exp. Syst., vol. 38, no. 7,       [38] A. Mittal and H. Gupta, ``An experimental evaluation in plant disease
       Nov. 2021, Art. no. e12746.                                                           identification based on activation-reconstruction generative adversarial
                                                                                             network,'' in Proc. 2nd Int. Conf. Advance Comput. Innov. Technol. Eng.
[18] H.-T. Thai, N.-Y. Tran-Van, and K.-H. Le, ``Artificial cognition for early              (ICACITE), Apr. 2022, pp. 361366.
       leaf disease detection using vision transformers,'' in Proc. Int. Conf. Adv.
       Technol. Commun. (ATC), Oct. 2021, pp. 3338.                                  [39] Q. Liang, S. Xiang, Y. Hu, G. Coppola, D. Zhang, and W. Sun,
                                                                                             ``PD2SE-Net: Computer-assisted plant disease diagnosis and severity esti-
[19] A. Khamparia, G. Saini, D. Gupta, A. Khanna, S. Tiwari, and                             mation network,'' Comput. Electron. Agricult., vol. 157, pp. 518529,
       V. H. C. de Albuquerque, ``Seasonal crops disease prediction and classifi-            Feb. 2019.
       cation using deep convolutional encoder network,'' Circuits, Syst., Signal
       Process., vol. 39, no. 2, pp. 818836, Feb. 2020.                              [40] R. Reedha, E. Dericquebourg, R. Canals, and A. Hafiane, ``Transformer
                                                                                             neural network for weed and crop classification of high resolution UAV
VOLUME 11, 2023                                                                              images,'' Remote Sens., vol. 14, no. 3, p. 592, Jan. 2022.

                                                                                                                                                                                     45391
                                                                                       A. Tabbakh, S. S. Barpanda: Deep Features Extraction Model

[41] S. Wu, Y. Sun, and H. Huang, ``Multi-granularity feature extraction based         [64] J. Shijie, H. Siping, and L. Haibo, ``Automatic detection of tomato disease
       on vision transformer for tomato leaf disease recognition,'' in Proc. 3rd              and pests based on leaf images,'' in Proc. Chin. Autom. Congr., 2017,
       Int. Academic Exchange Conf. Sci. Technol. Innov. (IAECST), Dec. 2021,                 pp. 25102537.
       pp. 387390.
                                                                                       [65] J. Chen, W. Wang, D. Zhang, A. Zeb, and Y. A. Nanehkaran, ``Atten-
[42] Y. Borhani, J. Khoramdel, and E. Najafi, ``A deep learning based approach                tion embedded lightweight network for maize disease recognition,'' Plant
       for automated plant disease classification using vision transformer,'' Sci.            Pathol., vol. 70, no. 3, pp. 630642, Apr. 2021.
       Rep., vol. 12, no. 1, pp. 110, Jul. 2022.
                                                                                       [66] J. Chen, D. Zhang, A. Zeb, and Y. A. Nanehkaran, ``Identification of
[43] S. Haug and J. Ostermann, ``A crop/weed field image dataset for the                      Rice plant diseases using lightweight attention networks,'' Exp. Syst. Appl.,
       evaluation of computer vision based precision agriculture tasks,'' in Proc.            vol. 169, May 2021, Art. no. 114514.
       Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, 2014, pp. 105116.
                                                                                       [67] P. S. Thakur, P. Khanna, T. Sheorey, and A. Ojha, ``Vision transformer
[44] L. Windrim, A. Melkumyan, R. J. Murphy, A. Chlingaryan, and                              for plant disease detection: PlantViT,'' in Proc. 6th Int. Conf. Cham,
       R. Ramakrishnan, ``Pretraining for hyperspectral convolutional neural net-             Switzerland: Springer, 2022, pp. 501511.
       work classification,'' IEEE Trans. Geosci. Remote Sens., vol. 56, no. 5,
       pp. 27982810, May 2018.                                                        [68] D. Suri, S. Saksenaa, U. Sehgal, and R. Garg, ``Disease classification in
                                                                                              wheat from images using CNN,'' in Proc. 13th Int. Conf. Cloud Comput.,
[45] C. D. Gurkaynak and N. Arica, ``A case study on transfer learning in                     Data Sci. Eng., Jan. 2023, pp. 566571.
       convolutional neural networks,'' in Proc. 26th Signal Process. Commun.
       Appl. Conf. (SIU), May 2018, pp. 14.                                           [69] Z. Lin, ``A unified matrix-based convolutional neural network for fine-
                                                                                              grained image classification of wheat leaf diseases,'' IEEE Access, vol. 7,
[46] X. Xia, C. Xu, and B. Nan, ``Inception-v3 for flower classification,'' in                pp. 1157011590, 2019.
       Proc. 2nd Int. Conf. Image, Vis. Comput. (ICIVC), 2017, pp. 783787.
                                                                                       [70] V. Kukreja and D. Kumar, ``Automatic classification of wheat rust diseases
[47] C. Narvekar and M. Rao, ``Flower classification using CNN and transfer                   using deep convolutional neural networks,'' in Proc. 9th Int. Conf. Rel.,
       learning in CNN--Agriculture perspective,'' in Proc. 3rd Int. Conf. Intell.            INFOCOM Technol. Optim. (ICRITO), Sep. 2021, pp. 16.
       Sustain. Syst. (ICISS), Dec. 2020, pp. 660664.
                                                                                       [71] W. Haider, A.-U. Rehman, N. M. Durrani, and S. U. Rehman,
[48] H. Hu, Z. Zhang, Z. Xie, and S. Lin, ``Local relation networks for                       ``A generic approach for wheat disease classification and verification
       image recognition,'' in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV),                 using expert opinion for knowledge-based decisions,'' IEEE Access, vol. 9,
       Oct. 2019, pp. 34643473.                                                              pp. 3110431129, 2021.

[49] F. Long, Z. Qiu, Y. Pan, T. Yao, J. Luo, and T. Mei, ``Stand-alone inter-                                        AMER TABBAKH received the B.Tech. degree in
       frame attention in video models,'' in Proc. IEEE/CVF Conf. Comput. Vis.                                        informatics engineering from Aleppo University,
       Pattern Recognit. (CVPR), Jun. 2022, pp. 31923201.                                                            Syria, in 2014, and the M.Tech. degree in computer
                                                                                                                      science and engineering from KIIT University,
[50] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,                                                Bhubaneswar, India, in 2020. He is currently a
       T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszko-                                      Ph.D. Scholar with the School of Computer Sci-
       reit, and N. Houlsby, ``An image is worth 1616 words: Transformers for                                        ence and Engineering, VIT-AP University, Amar-
       image recognition at scale,'' 2020, arXiv:2010.11929.                                                          avati, India. His research interests include machine
                                                                                                                      learning, deep learning, and image processing.
[51] Kaggle. The Dataset is Taken From the Kaggle Opensource Link.
       Accessed: Aug. 2022. [Online]. Available: https://www.kaggle.com/                                              SOUBHAGYA SANKAR BARPANDA received
       datasets/emmarex/plantdisease                                                                                  the B.Tech. degree in computer science and
                                                                                                                      engineering from the Biju Patnaik University of
[52] D. Vasan, ``IMCFN: Image-based malware classification using fine-tuned                                           Technology, Rourkela, and the M.Tech. degree
       convolutional neural network architecture,'' Comput. Netw., vol. 171,                                          in software engineering and the Ph.D. degree
       Apr. 2020, Art. no. 107138.                                                                                    from NIT, Rourkela. He is currently an Associate
                                                                                                                      Professor with the School of Computer Science
[53] A. Krizhevsky, I. Sutskever, and G. E. Hinton, ``ImageNet classification                                         and Engineering, VIT-AP University, India. His
       with deep convolutional neural networks,'' Commun. ACM, vol. 60, no. 2,                                        research interests include image processing, soft-
       pp. 8490, Jun. 2012.                                                                                          ware engineering, and biometrics.

[54] A. Fuentes, S. Yoon, S. Kim, and D. Park, ``A robust deep-learning-                                                                                                                VOLUME 11, 2023
       based detector for real-time tomato plant diseases and pests recognition,''
       Sensors, vol. 17, no. 9, p. 2022, Sep. 2017.

[55] S. C. Bhardwaj, P. Prasad, O. P. Gangwar, H. Khan, and S. Kumar, ``Wheat
       rust research-then and now,'' Indian J. Agric. Sci., vol. 86, pp. 12311244,
       Oct. 2016.

[56] K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image
       recognition,'' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR),
       Jun. 2016, pp. 770778.

[57] S. Coulibaly, B. Kamsu-Foguem, D. Kamissoko, and D. Traore, ``Deep
       neural networks with transfer learning in millet crop images,'' Comput.
       Ind., vol. 108, pp. 115120, Jun. 2019.

[58] K. Simonyan and A. Zisserman, ``Very deep convolutional networks for
       large-scale image recognition,'' 2014, arXiv:1409.1556.

[59] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, ``Rethinking
       the inception architecture for computer vision,'' in Proc. IEEE Conf.
       Comput. Vis. Pattern Recognit. (CVPR), Jun. 2016, pp. 28182826.

[60] C. Feng, H. Zhang, S. Wang, Y. Li, H. Wang, and F. Yan, ``Structural
       damage detection using deep convolutional neural network and transfer
       learning,'' KSCE J. Civil Eng., vol. 23, no. 10, pp. 44934502, Oct. 2019.

[61] R. M. Mohana, C. K. K. Reddy, and P. R. Anisha, ``A study and early iden-
       tification of leaf diseases in plants using convolutional neural network,'' in
       Proc. 4th Int. Conf. Smart Comput. Informat. Cham, Switzerland: Springer,
       2021, pp. 693709.

[62] G. Sachdeva, P. Singh, and P. Kaur, ``Plant leaf disease classification using
       deep Convolutional neural network with Bayesian learning,'' Mater. Today,
       Proc., vol. 45, pp. 55845590, Jan. 2021.

[63] M. A. Jasim and J. M. Al-Tuwaijari, ``Plant leaf diseases detection and
       classification using image processing and deep learning techniques,'' in
       Proc. Int. Conf. Comput. Sci. Softw. Eng. (CSASE), Apr. 2020, pp. 259-
       265.

45392


---

## EXTRACTED FIGURES (visual analysis)

Rendered pages: figures/page6.png, page7.png, page8.png, page9.png (under papers/20_TLMViT/inputs/figures/).

### Figure 2 - Architecture of TLMViT (page 6, `papers/20_TLMViT/inputs/figures/page6.png`)
Left to right: Input 256*256*3 -> "Pre-Trained Model" (conv layers ... output 8*8*C) [Initial Feature Extraction] -> Partitioning into 4 patches each 4*4*C -> Flatten Patches (1*4*4*C each) -> Linear Projection -> + Position embeddings (0..4), plus Class Embedding token (Z0); Z1..Z4 -> Transformer Encoder -> MLP Head -> C1..C15 [Classification]. Diagram labels only; numeric values match the text (patch 4x4, 4 patches + cls token = 5 tokens, proj dim 64).

### Table 1 - PlantVillage subset (page 6): 15 classes, total 20,638
Tomato mosaic virus 373; Tomato Yellow Leaf Curl Virus 3208; Tomato Target Spot 1404; Tomato Septoria leaf spot 1771; Tomato Early blight 1000; Tomato Leaf Mold 952; Tomato Spider mites 1676; Tomato Late blight 1909; Tomato Bacterial spot 2127; Tomato healthy 1591; Pepper bell Bacterial spot 997; Pepper bell healthy 1478; Potato Late blight 1000; Potato healthy 152; Potato Early blight 1000.

### Table 2 - Wheat Rust (page 6): Brown Rust 1128; Yellow Rust 1348; Healthy 1203; total 3679.

### Table 3 - AlexNet feature map sizes (page 7) [layer, filter, stride, default 227 input, our 256 input]
Input - - 227*227*3 | 256*256*3
Conv1 11*11 s4: 55*55*96 | 64*64*96
MaxPool1 3*3 s2: 27*27*96 | 32*32*96
Conv2 5*5 s1: 27*27*256 | 32*32*256
MaxPool2 3*3 s2: 13*13*256 | 16*16*256
Conv3 3*3 s1: 13*13*384 | 16*16*384
Conv4 3*3 s1: 13*13*384 | 16*16*384
Conv5 3*3 s1: 13*13*256 | 16*16*256
MaxPool3 3*3 s2: 6*6*256 | 8*8*256
(Paper's table values as printed; some sizes are inconsistent with strides, e.g. 227->55 for default, transcribed literally.)

### Table 4 - VGG16 / VGG19 (page 7), columns: layer, filter, default 224 input, PlantVillage 256 input
VGG16: Input 224*224*3 | 256*256*3; 2 Conv 3*3*64: 224*224*64 | 256*256*64; MaxPool s2: 112*112*64 | 128*128*64; 2 Conv 3*3*128: 112*112*128 | 128*128*128; MaxPool s2: 56*56*128 | 64*64*128; 3 Conv 3*3*256: 56*56*256 | 64*64*256; MaxPool s2: 28*28*256 | 32*32*256; 3 Conv 3*3*512: 28*28*512 | 32*32*512; MaxPool s2: 14*14*512 | 16*16*512; 3 Conv 3*3*512: 14*14*512 | 16*16*512; MaxPool s2: 7*7*512 | 8*8*512. Total layers 16.
VGG19: identical but 4 Conv in the 256, 512, 512 blocks (2,2,4,4,4); total layers 19; final 7*7*512 | 8*8*512.
(Small print; read at zoom 3, legible.)

### Table 5 - Experimental parameters (page 9)
Image augmentation: rotation random in [-25, 25] deg; width shift 10%; height shift 10%; shear 20%; horizontal and vertical flip TRUE; zoom 20%.
Dataset split: Training 80%; Validating and Testing 20% (~4100 images).
Training: Batch size 32; Epochs 25; Optimizer row reads "RMSprop  adam" (inconsistent in paper; text says Adam is used); Initial LR 1.00E-03.
Classifier MLP: hidden1 1024, hidden2 512 (input 256, output 15).
ViT: patch size 4; projection dim 64; num heads 4; transformer layers 8; transformer MLP = two hidden layers 128 then 64 (text); LayerNorm "Norm"; residual connections (eq. 5-6); loss categorical cross-entropy.

### Key shapes (from text, page 8)
Initial features 8*8*C: AlexNet 256, VGG16 512, VGG19 512, ResNet50 2048 (256 input), InceptionV3 2048 (images rescaled to 299*299*3 -> 8*8*2048; note text also writes R^{6x6x2048} for Inception). Wheat Rust images rescaled to same as PlantVillage.
Results (Table 6, val acc ViT-based): ResNet50 89.2, AlexNet 90.49, VGG16 98.43, VGG19 98.81, InceptionV3 98.48; train acc 90.405, 92.5, 99.60, 98.8, 99.2. TL-only val acc: 79.84, 81.91, 89.31, 86.55, 90.77.
Tables 6-12 and Figs 6-8 (results/confusion matrices) were not rendered; their text is in the body above.

### Table 8 / Table 9 - F1 / Precision / Recall (page 11, `papers/20_TLMViT/inputs/figures/page11.png`)
Table 8 (PlantVillage) — [TL-only F1, Prec, Rec] | [TLMViT F1, Prec, Rec]:
ResNet50 79.56/79.42/79.71 | 89.56/89.38/89.75; AlexNet 81.78/81.64/81.94 | 90.31/90.11/90.53; Inception V3 90.26/90.39/90.15 | 98.32/98.36/98.29; VGG16 89.24/89.32/89.18 | 98.23/98.15/98.32; VGG19 86.4/86.37/86.44 | 98.73/98.72/98.76.
Table 9 (Wheat Rust) — ResNet50 82.99/82.79/83.21 | 93.13/92.99/93.27; AlexNet 83.23/83.31/83.16 | 93.57/93.48/93.67; Inception V3 93.36/93.41/93.21 | 99.92/99.93/99.91; VGG16 91.81/91.87/91.75 | 99.58/99.65/99.51; VGG19 88.42/88.49/88.36 | 99.71/99.78/99.65.
Figure 7 (same page) = learning curves of TL-only models (25 epochs).
