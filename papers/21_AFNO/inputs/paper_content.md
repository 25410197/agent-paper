# PAPER CONTENT: 21_AFNO

**Source PDF:** papers/21_AFNO/inputs/fpls-14-1231903.pdf
**Extracted:** 2026-10-20

---

                                                   TYPE Original Research
                                                   PUBLISHED 12 September 2023
                                                   DOI 10.3389/fpls.2023.1231903

OPEN ACCESS                                    FOTCA: hybrid transformer-CNN
                                               architecture using AFNO for
EDITED BY                                      accurate plant leaf disease
                                               image recognition
Muhammad Fazal Ijaz,
Melbourne Institute of Technology,             Bo Hu1, Wenqian Jiang2, Juan Zeng3, Chen Cheng4
Australia                                      and Laichang He2*

REVIEWED BY                                    1School of Information Engineering, Nanchang University, Nanchang, China, 2Department of
                                               Radiology, the First Affiliated Hospital of Nanchang University, Nanchang, China, 3Second Clinical
Yongqiang Zheng,                               Medical College, Nanchang University, Nanchang, China, 4School of Mathematics and Statistics,
Southwest University, China                    Huazhong University of Science and Technology, Wuhan, China
Nisha Pillai,
Mississippi State University, United States    Plants are widely grown around the world and have high economic benefits.
                                               plant leaf diseases not only negatively affect the healthy growth and
*CORRESPONDENCE                                development of plants, but also have a negative impact on the environment.
                                               While traditional manual methods of identifying plant pests and diseases are
Laichang He                                    costly, inefficient and inaccurate, computer vision technologies can avoid these
    laichang_he@163.com                        drawbacks and also achieve shorter control times and associated cost
                                               reductions. The focusing mechanism of Transformer-based models(such as
RECEIVED 01 June 2023                          Visual Transformer) improves image interpretability and enhances the
ACCEPTED 21 August 2023                        achievements of convolutional neural network (CNN) in image recognition, but
PUBLISHED 12 September 2023                    Visual Transformer(ViT) performs poorly on small and medium-sized datasets.
                                               Therefore, in this paper, we propose a new hybrid architecture named FOTCA,
CITATION                                       which uses Transformer architecture based on adaptive Fourier Neural
                                               Operators(AFNO) to extract the global features in advance, and further down
Hu B, Jiang W, Zeng J, Cheng C                 sampling by convolutional kernel to extract local features in a hybrid manner. To
and He L (2023) FOTCA: hybrid                  avoid the poor performance of Transformer-based architecture on small
transformer-CNN architecture                   datasets, we adopt the idea of migration learning to make the model have
using AFNO for accurate plant                  good scientific generalization on OOD (Out-of-Distribution) samples to improve
leaf disease image recognition.                the model's overall understanding of images. In further experiments, Focal loss
Front. Plant Sci. 14:1231903.                  and hybrid architecture can greatly improve the convergence speed and
doi: 10.3389/fpls.2023.1231903                 recognition accuracy of the model in ablation experiments compared with
                                               traditional models. The model proposed in this paper has the best
COPYRIGHT                                      performance with an average recognition accuracy of 99.8% and an F1-score
                                               of 0.9931. It is sufficient for deployment in plant leaf disease image recognition.
� 2023 Hu, Jiang, Zeng, Cheng and He. This
is an open-access article distributed under    KEYWORDS
the terms of the Creative Commons
Attribution License (CC BY). The use,          plant leaf disease image recognition, hybrid architecture, transformer-based models,
distribution or reproduction in other          adaptive Fourier Neural Operator, deep learning
forums is permitted, provided the original
author(s) and the copyright owner(s) are
credited and that the original publication in
this journal is cited, in accordance with
accepted academic practice. No use,
distribution or reproduction is permitted
which does not comply with these terms.

Frontiers in Plant Science                     01  frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                       10.3389/fpls.2023.1231903

1 Introduction                                                                           to traditional self-attention, AFNO maps images to
                                                                                         frequency domain for better performance.
    As one of the main sources of economic growth in many                            � A model architecture that integrates both global and local
developing countries, it is necessary to produce agricultural crops                      features has been proposed, connecting CNNs and
in a stable and efficient manner (Iqbal et al., 2018). However,                          Transformers through inter-level concatenation to achieve
the expansion of crops, combined with the overuse of pesticides                          coupling of global and local receptive fields. This hybrid
and the exacerbation of global climate change, has led to an increase                    architecture, which blends global and local features, can
in the occurrence and spread of agricultural pests and diseases.                         better utilize the extracted features and improve the
Therefore, controlling these pests and diseases is becoming                              performance and robustness of the model.
increasingly challenging. Early detection and treatment of such                      � This article proposes that using Focal Loss as the loss
pests and diseases have unique advantages (Sanju and VelammaL,                           function can effectively enhance the model's ability to
2021). In the early stages of pest infestation, it is difficult to                       train on difficult samples.
distinguish the leaves of affected plants from those of normal
because of the high interclass variation in colour and profile and                  The rest of this article is organized as follows. The Section 3
the low intraclass variation.                                                   mainly elucidates the details of the dataset composition and model
                                                                                composition structure, in Section 4 we specify the optimization
    Traditional methods for identifying agricultural pests typically            scheme for settings other than the model.
involve visual inspection of crops by farmers and agricultural
experts. However, these approaches are costly, time-consuming,                  2 Related works
highly subjective, and non-transferable (Ouppaphan, 2017).
Although some success has been achieved in classifying                          2.1 Convolutional Neural Networks
agricultural images using traditional image processing techniques,
these methods face several challenges. First, they often require                    Convolutional Neural Networks (CNNs) have become a
manual feature extraction, increasing workload. Second, the                     popular and powerful tool for image recognition in various fields.
manually extracted features may not adequately represent the                    In 1998, Lecun et al. (1998) pioneering introduction of the LeNet-5
characteristics of agricultural images, leading to semantic gaps.               model laid the fundamental framework for CNNs. In 2012, the
Lastly, the variability and complexity of agricultural images,                  AlexNet model (Krizhevsky et al., 2012) used more convolutional
combined with factors such as image quality and shooting angle,                 layers and a larger parameter space to fit large-scale datasets in
can significantly affect the final recognition results, rendering these         ILSVRC (ImageNet Large-Scale Visual Recognition Challenge). It
techniques unsuitable for large-scale applications (Bisen, 2021; Pan            was a groundbreaking demonstration of the advantages of deep
et al., 2022).                                                                  neural networks over shallow neural networks, achieving
                                                                                significantly higher accuracy than the runner-up in recognition
    In light of relevant research on plant leaf disease identification and      accuracy. This breakthrough established the status of CNNs in
fine-grained recognition, we investigated issues associated with the            computer vision and brought new opportunities for plant leaf
relatively coarse recognition of algorithms in current crop pest                disease identification.
identification approaches and their inadequate performance on
datasets containing multiple, similar pests and diseases. We                    2.2 Fine-grained image recognition
recognized the potential of the Transformer-based model and applied
it to this domain. However, using the original patch and position                   State-of-the-art deep learning algorithms for image recognition
embedding(Pap Embedding) would impose limitations on the                        predominantly rely on publicly available datasets such as ImageNet.
experiment outcomes. Concurrently, utilizing the adaptive Fourier               Although utilizing these datasets enhances a model's capability to
basis function to convert images to the frequency domain and                    identify objects in images, including contour features (Wang D. et
deploying a CNN-Transformer architecture to separately extract local            al., 2021; Zhang and Wang, 2016), discerning different types of pests
and global features could enhance the training upper limit. As a solution,      affecting the same plant leaves in plant leaf disease images remains
we propose a novel hybrid architecture for plant leaf disease image             challenging due to their similar contour features (Kong et al., 2021).
recognition, termed FOTCA (where F and O signify Adaptive Fourier               Merely transferring a model trained on other data categories to the
Neural Operator (AFNO) (Guibas et al., 2021) and TCA represents the             study of plant leaf disease might not yield the anticipated accuracy.
Transformer-CNN Architecture). This approach addresses and                      Consequently, models should be encouraged to learn fine-grained
optimizes the convergence issue and generalization capability of the            features of objects for improved results (Berg et al., 2014; Yang et al.,
ViT model, further improving training outcomes. Additionally, we                2018; Chen et al., 2019)
evaluate the performance of FOTCA using the Plant-village dataset, a
plant leaf disease dataset, to assess its scope and effectiveness.                  To tackle this issue, numerous researchers have explored
                                                                                convolutional neural networks in fine-grained recognition. For
    In summary, this work makes the following three                             instance, Zhang et al. (2014) developed a model that surmounts
main contributions.                                                             these constraints by utilizing deep convolutional features derived

     � This article applies an operator called Adaptive Fourier
         Neural Operator (AFNO) and learnable Fourier features
         which can replace traditional position encoding. Compared

Frontiers in Plant Science                                                  02  frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                         10.3389/fpls.2023.1231903

from bottom-up region proposals for fine-grained recognition.                  (2019) improved the loss function in order to enable the
Multi-proposal Net (Zhang et al., 2016) obtains image blocks                   classification model to learn features with greater distinction
through the Edge Box Crop method and incorporates an output                    between more difficult to distinguish classes, and used feature map
layer of key points and visual features to further reinforce the local         inhibition methods to enable the model to learn subtle differences.
feature positional relationship between local features and overall
information. Deep LAC (Lin et al., 2015) performs part localization,           3 Materials and methods
alignment and classification in the same network, and the VLF
(valve linkage function) function is proposed for back propagation             3.1 Dataset and data pre-processing
in Deep LAC, which is able to adaptively reduce the classification
and alignment errors and update the localization results.                          In this study, the base dataset was selected from the publicly
                                                                               available plant-village dataset on the web. This is a dataset
2.3 Transformer-based models for fine-                                         specifically designed to study the work of various plant leaf
grained visual recognition                                                     disease recognition models, with a total of 54,303 images
                                                                               containing a total of 38 different species of 13 plant species
    As a decoding and encoding architecture model based on                     (including apple, blueberry, cherry, corn, grape, orange, bell
attention mechanism, Transformer (Vaswani et al., 2017) has been               pepper, potato, raspberry, soybean, pumpkin, strawberry, tomato).
widely applied in the field of natural language processing. Inspired by        Plant images, including non-plant leaf species, were inserted to train
this significant achievement, many scholars have transferred the               the model to recognize non-plant leaf images. The ratio of the
Transformer structure to computer vision tasks and have achieved               training and validation sets was 8:2.
considerable success (Carion et al., 2020; Dosovitskiy et al., 2020;
Wang Y. et al., 2021). The attention mechanism primarily identifies                Before inputting the images into the model, the images need to
and classifies objects through different parts of an object, thus making       be quantified uniformly. In the first step, the training set is
it possible to focus on key features of the subject for datasets of plant      expanded by applying some image enhancement techniques to
leaf diseases with small differences between inter-class. As a result,         each image independently superimposed, and the main data
the Transformer structure completes more successful recognition                enhancement methods used in this study are RandomFlip,
compared to CNN structures. Focusing on this area of study, Cai et al.         RandomCrop, RandomHorizontalFlip and RandomResizeCrop,
(2021) introduced a ViT with adaptive attention that adds attention            mainly to eliminate the shift of the final recognition result by the
weakening and strengthening modules. This improves the                         change of the shooting angle in real life, and to prevent the neural
performance of key features while capturing more feature                       network from overfitting phenomenon. In addition, the image size
information. Fu et al. (2017) used multiple scales of recurrent                is uniformly adjusted to a square of size 224*224 pixels to facilitate
attention links to learn the target feature region and used within-            further processing by the model. Figure 1 shows a portion of the
scale classification loss and between-scale ranking loss to make the           dataset images and the pre-processing process of the images.
model more focused on the finer-grained features of the target object.
He et al. (2022) proposed the TranFG framework, which integrates               3.2 Model
all original attention weights of the Transformer into one attention
map, enabling the model to recognize image blocks and calculate                    In this paper, we study applying deep transfer learning to all
their relationships while utilizing contrastive loss to expand the             models in the model and comparison experiments. The Pre-train
distance between confusion class feature representations. Sun                  and Fine-tuning approach in deep transfer learning is the most

                            A                                                  B

FIGURE 1

Plant Village dataset presentation and data augmentation examples. (A) A partial sample images of the dataset (containing healthy and sick). (B) Data
augmentation operation examples.

Frontiers in Plant Science                                                 03     frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                                                                                10.3389/fpls.2023.1231903

convenient and reliable method for deep model learning. By putting                corresponding models, effectively speeding up the training process
a pre-trained model with some generalization ability onto a new                   of the models by means of transfer learning.
similar dataset, the model will show superior performance than
training a model from scratch after a simple and short training                       The FOTCA model studied in this article is mainly composed of
process. Similarly, the pre-trained model is adjusted for a specific              a shortcut module and a Transformer module. The proposed overall
dataset by a loss calculation function called model fine-tuning, and              model structure is shown in the Figure 2A, which is mainly divided
the loss calculation function widely used in deep neural networks                 into three steps: Patch and Position Embedding, Transformer
today is the CrossentropyLoss (CE) (Shown in Figure 2B), which is                 architecture based on adaptive Fourier Neural Operators
calculated as follows:                                                            and Classifier.

           Loss = L(y, p^) = -y log (p^) - (1 - y) log (1 - p^)  (1)              3.2.1 Patch and position embedding

                                                                                        The module consists of two sub-modules: Patch Embedding

    where p^ is the predicted probability and y is the actual outcome             and Positional encoding based on learnable Fourier features.
category. A remarkable feature of this loss calculation is that the losses
are also treated consistently for simple and easily scored samples ( pt           Compared with the regular position embedding, a primary
 0:5). This result may also occur when the losses of a large number
of simple samples accumulate and the small loss may swamp the sparse              advantage of Positional Embedding based on learnable Fourier
classes, or when there are differences in the loss of different classes in a
near-saturated training process. There are two intuitive ways to deal             features is that it provides richer positional information,
with this type of problem, one is to add weighting factors directly to the
loss function, and the other is to add adjustable factors represented by          especially for input from large datasets and long sequences. It can
focal loss. In this paper, focal loss is introduced into the model by
reducing the loss function to the weights of the easy samples, which can          more finely encode each position through learnable Fourier basis
help the loss function to favour the difficult samples and improve the
accuracy of the difficult samples (Shown in Figure 2C). The calculation           functions, and this learnability makes it more flexible and adaptable.
procedure is as follows:
                                                                                  Additionally, it has better local connectivity and translational

                                                                                  invariance when performing convolution operations, which can

                                                                                  further enhance the model's performance.

                                                                                        For the input image X  RH�W�C, Assuming we use a patch size of
                                                                                  ( ) ( ) P � P, we can divide the input image into
                                                                                                                                     W     �       H  local regions
                                                                                                                                     P             P
                                                                                          Rp�pmesc                                         �1,     , j    �1,     ).
                                                                                  Pi,j              according  to the  patch  size.  (i         H              W
                                                                                                                                                P              P

                                                                                  And expand it into a one-dimensional vector vi,j, whose size is D, where

           Lossfl = -(1 - pt)g log (pt)                                           D is an adjustable parameter representing the vector dimension after

                                                                 (2)              patch embedding. For each matrix, we use a learnable weight matrix

    where pt reflects the proximity to the ground truth. Larger pt                Wwatch of size D � K to map it to a low dimensional space:
means closer to category y, representing more accurate recognition.
g is the adjustable factor.                                                                         xip,jatch = Wpatch � vi,j                                     (3)

    Due to the wide recognition of the ImageNet dataset, as well as                   There, xip,jatch  RD�K is the feature vector obtained through
the excellent performance and superior model fitting ability of the               patch embedding.
pre-trained models, all the experiments cited in this paper are based
on the pre-trained models of the ImageNet dataset loaded with the                       Next, we generate position encoding vectors for each patch. For

                                                                                  each coordinate binary (i, j), use a learnable Fourier feature function

                            A                                                     B

                                                                                  C

FIGURE 2

FOTCA model architecture and two loss functions for fine-tuning (A) The figure provides an overview of the detailed structure of the entire FOTCA
model's neural network. (B) Cross-Entropy loss for fine-tuning. (C) Focal loss for fine-tuning.belflow chart.

Frontiers in Plant Science                                                    04                                                                      frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                                                                                                       10.3389/fpls.2023.1231903

to generate a vector fip,jos with a size of 2 K:                                                     Then, by performing global average pooling on A, obtain the weight
                                                                                                     wk of each basis function, and the calculation formula is:
           fip,jos(p) = �cos (p, w1 + b1), sin (p, w1 + b1), ...,
                                                                                            (4)
                                                                                                                                          1              H=PW=P
                    cos (p, wK  + bK ), sin (p, wK  + bK )                                                                                �
                                                                                                                                                                 ai,j,k

                                                                                                                                                         i=1 j=1
                                                                                                                       o o wk=                                                         (9)
                                                                                                                              (H=P)           (W=P)
                    (i-1)     (j-1)
Where      p  =  (   H     ,   W     )   is  the  normalized                representation  of

input coordinates, wK and bK are learnable parameters, and �, �                                          The original data is ultimately mapped to the frequency domain
                                                                                                     through basis functions and basis function weights.
represents inner product between vectors. Finally, we add the

position encoding vector to the feature vector obtained from                                                                              K
patch embedding, resulting in the final feature vector xip,jos:                                                               o Xi,j = wk � Bi,j,k,: � xi,j,:
                                                                                                                                                                                       (10)

                                                                                                                                         k=1

                              xip,jos = xip,jatch + fip,jos                                 (5)      Where Xi,j,: denotes the flattened vector of patch (i,j). Finally, we

    Then, we can use encoders to encode this sequence for further                                    merge the frequency-domain results of all patches into a large
processing and recognition tasks.
                                                                                                     frequency-domain      tensor      F      C  H  �WP  �k  .  For  channel  mixing,  it  is
                                                                                                                                                 P

                                                                                                     done through an MLP on this frequency-domain tensor to extract

3.2.2 Adaptive Fourier Neural Operators                                                              high-level features.

    This module maps the input image to the frequency domain                                         Zi(,kj )  =  MLP               i               j             W      k    �1,  N   (11)
and uses adaptive Fourier basis functions to map each small block in                                                    Fi(,kj ) ,        1,  H               1,  P   ,
the time domain to the frequency domain. This allows better                                                                                   P,
extraction of frequency domain features and offers advantages in
processing periodic as well as regular images, while providing better                                Finally, we concatenate all frequency domain features z together
robustness to transformations such as rotation and scaling of the
input data. Specifically, for each small block x, its frequency-domain                               to form the final feature vector.
representation xfreq is obtained by first mapping it from the time
domain to the frequency domain through Fast Fourier Transform                                                                          h                          i
(FFT). The (i, j)th element represents the frequency-domain                                                                   Z = z(1), z(2), ..., z(F)
representation at position (i, j) on the frequency domain.                                                                                                                             (12)

                                                                                                         The feature vector can be fed into any type of neural network for
                                                                                                     further training or inference operations. This process has strong
                                                                                                     expressiveness and interpretability in data processing.

                                     P-1     P-1                mPi+nPj                              3.2.3 Classifier
                    o o Xfi,rjeq                      e-2p  i(           )
                              =                 Xm,n                                        (6)          When using the Transformer-based model, its output vector is
                                                                                                     often passed to a classifier for image recognition. The traditional
                                     m=0n=0                                                          approach is to use the hidden layers of MLP to collect features and
                                                                                                     perform classification. However, there are many localized feature
    Next, we will map each patch from the time domain to the                                         points in plant leaf disease images, and these localized feature points
frequency domain using adaptive Fourier basis functions.                                             can usually be extracted and represented more efficiently by means
Specifically, for each small block Xi,j, we calculate the inner                                      of convolutional operations. Therefore, the use of CNNs may be
product between it and the adaptive Fourier basis functions, and                                     more appropriate for such problems. Here, we consider using a
use the inner product value as coefficients.                                                         basic block as a classifier, which includes convolutional layers and
                                                                                                     batch normalization layers, as well as shortcut operations. By
                                     1    pp                                                         upsampling to a higher dimension to obtain local feature values,
                                     p2                                                              feature transfer and information fusion are achieved. Also, shortcut
                                               Xi,j(u, v)Bu,v,k                                      operations can avoid gradient disappearance and model overfitting
                                                                                                     problems caused by excessive stacking of convolutional layers.
                                         u=1v=1
                    o o ai,j,k=                                                             (7)          After downsampling or upsampling the feature maps using
                                                                                                     convolutional layers, the downsampled or upsampled feature maps
    The coefficient of small block Xi,j under the kth adaptive Fourier                               are added to the output feature map using element-wise addition,
basis function is ai,j,k, p is the size of the small block, and Bu,v,k is the                        which implements shortcut. This operation allows the model to
value of the kth adaptive Fourier basis function at position (u, v).                                 better learn details and local features in the input, thereby
                                                                                                     improving the model's performance. In this article's classifier
The specific calculation formula is as follows.                                                      design, a 1 � 1 convolutional layer is embedded in the shortcut to
                                                                                                     adjust the depth of the feature maps so they can be added to the
              Bu,v,k = cos (wkT vu,v) + i � sin (wkT vu,v)                                  (8)      feature maps from the next layer. This helps preserve lower-level
                                                                                                     features, enhance inter-channel communication, and improve the
    Where vu,v denotes the spatial location vector of the center point
of the (u, v)th patch in the input image and wk represents the
adaptive basis vector.

    Expand all small adaptive Fourier basis functions into a large
adaptive Fourier basis function matrix B  Cp2�K , where K is the
number of adaptive Fourier basis functions. At the same time, expand

all small adaptive Fourier coefficients into a large adaptive Fourier
coefficient matrix A  CN�K , where N is the number of small blocks.

Frontiers in Plant Science                                                                       05                                                                           frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                                                                        10.3389/fpls.2023.1231903

model's performance. Specifically, the input vector undergoes two                     according to the number of learning rate updates to linearly
3 � 3 convolutional layers consecutively, with a certain amount of                    decay on top of the original one. The loss calculation function of
non-linear activation function (such as ReLU) inserted between                        the FOTCA model uses focal loss, and the rest of the models in the
them, as shown below:                                                                 comparison experiments use the cross-entropy loss function.
                                                                                      Dropout regularization is also used in the training process. By
           x1 = ReLU(Conv3�3(x))   (13)                                               randomly dropping some neuron connections temporarily during
                                                                                      the training process, the purpose is to effectively avoid overfitting of
           x2 = ReLU(Conv3�3(x1))  (14)                                               the model during the training process. The generalization ability of
                                                                                      the model is also enhanced. The specific hyperparameter settings
    At the same time, downsampling is applied to the input vector                     for the experiments are shown in Table 1.
to promote the underlying features towards the final recognition
result. This operation can be achieved by using a 1 � 1 convolution                       We choose accuracy, adjustment time for accuracy, loss,
layer in the shortcut. Finally, add the output of twoconsecutive                      adjustment time for loss, F1-score, parameters and FLOPS. We
convolution layers to the output of the shortcut, as shown below:                     define for the first time the adjustment time for accuracy (loss).
                                                                                      Adjustment time is the number of iterations required during model
                 xout = ReLU (x2 + xshortcut)                                         training to bring the model performance from the initial
                                                                            (15)      performance to 95% difference from the final converged
                                                                                      performance. This metric reflects the ability of the model in terms
                      = ReLU (x2 + ReLU (Conv1�1(x)))                                 of speed of convergence as well as training efficiency. The concept of
                                                                                      adjustment time helps to deeply evaluate and compare the speed of
    Among them, xshortcut is the output of the shortcut, and it needs                 the model training process. A shorter adjustment time means that
to ensure that its depth is the same as the output depth of the second                the model reaches its potential performance faster within a limited
convolution layer.                                                                    number of iterations. Thus, with this metric, researchers can more
                                                                                      intuitively assess the performance gap between different models.
    In the Classifier of this article, it has been demonstrated that
shortcuts effectively alleviate the gradient vanishing problem,                           In addition, adjustment time complements other commonly
resulting in a decent convergence rate even when applied to very                      used evaluation metrics (e.g., accuracy, loss, etc.) to provide
deep models. Using two 3 � 3 convolutional layers instead of a                        researchers with a more comprehensive view of how a model
larger kernel increases nonlinearity, allows for faster collection and                performs during training. In the case of similar model
extraction of local features, and reduces the number of parameters,                   performance, shorter adjustment times may be an advantage due
making the model easier to train and generalize.                                      to greater savings in computational resources and time.

4 Experiments and discussion

4.1 Experimental environment                                                          4.3 Comparative experiments

    The models we studied was developed based on the open source                          Firstly, the performance of Vision Transformer (ViT) is
of deep learning framework pytorch1.11.0 with the following                           typically affected by the size of the training dataset due to the fact
experimental equipment: CPU is Intel(R) Xeon(R) Platinum                              that ViT is based on transformers methods, which typically require
8255C, GPU is single card V100-SXM2-32GB, CUDA version is                             a large amount of data for effective training. This is because
11.3, and programming language is Python3.8 (ubuntu20.04).                            transformers models, including ViT, are variants of the self-
                                                                                      attention mechanism, which allows the model to capture the

4.2 Experimental parameters,                                                          TABLE 1 Hyperparameter tuning for the model.
evaluation indicators

    Based on earlier work by scholars (Kumar et al., 2023a; Kumar                     Initial learning rate                              0.001
et al., 2023b; Wei et al., 2023), we selected DenseNet-169, Inception-
v3, VGG19, ResNet-50, ResNet-101, ViT as our comparative                              Epochs                                             100
experimental models to verify the feasibility and reliability of the
improved model plant leaf disease image recognition method in this                    Batch size                                         8
paper. We all iteratively update the pre-trained model parameters
based on each model to accelerate the model convergence during                        Image size(for all)                                224*224
the training process.
                                                                                      Image size(for Inception-v3)                       299*299
    The training process uniformly uses the stochastic gradient
descent SGD optimization algorithm to optimize its model. The                         Transformer-based models      Embedding dimension  768
same learning rate adjustment strategy is used for each parameter in
the model, and the learning rate is dynamically adjusted using                                                      Patch size           16
LambdaLR, which is adjusted by adjusting the learning rate
                                                                                                                    Head number          8

                                                                                                                    Depth                12

                                                                                      This table presents the various hyperparameters and their selected values used of the proposed
                                                                                      FOTCA model and compared models.

Frontiers in Plant Science                                                        06                                                     frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                   10.3389/fpls.2023.1231903

global relationships of the input data, but this mechanism also             and data characteristics. In many cases, if feasible, using larger
requires a large amount of data to support.                                 datasets often yields better results.

    For the specific minimum sample requirement, it may vary as                 Next, we compared six mainstream image recognition models,
it depends on several factors, including the size of the model (e.g.,       using the same backbone training model and dataset to ensure the
the number of layers of the model, the number of hidden units, etc.),       fairness of testing. The only exception is that the input image size
the complexity of the task (Lee et al., 2021) and the distribution of       must be 299 � 299 for the Inception-v3 model. We show the
the data. Therefore, it can only be explored by trying the following        accuracy and loss values of each model during the iteration
experimental procedure.                                                     process in the chart. Due to the significant difference between the
                                                                            initial and final iteration results, the observation of the later
    We chose to compare the ViT model with the ResNet101                    iteration effect is not very clear and accurate. Therefore, we re-
model. This decision was made because ResNet101 has similar                 drew the chart showing the changes in the accuracy and loss values
flops and comparable convergence capability to ViT on large-scale           of later iterations between 20 and 100 iterations to achieve deeper
datasets. In preliminary experiments, we selected training set              analysis and optimization. This approach can enhance our
proportions of 6%, 8%, 10%, 20%, and 80%, respectively, to                  understanding of the model's performance, which helps us
investigate the convergence capability and accuracy of ViT and              further improve and optimize our algorithms. The final
ResNet101 on medium-sized datasets. The experimental results are            experimental effect comparison is shown in Figure 4. Specifically
shown in Figures 3A�C.                                                      shown in Figure 3D. The plot offers a visual representation of the
                                                                            balance between performance and computational efficiency among
    We observed that ViT's training performance deteriorated                different models. The final result of the iteration is shown in Table 2.
rapidly on medium-sized datasets when the size of the training
set was small. In contrast, ResNet101 demonstrated smaller                      According to the chart, the six mainstream image recognition
variations and more stable curves, with minimal decreases in                models showed good feature extraction and representation abilities
accuracy. Particularly, there was a significant turning point in            in the first 20th epochs, with an accuracy of over 95% and a loss of
ViT's performance when the training set was at 8%. However,                 below 0.35 for all of them. However, we observe a strange status quo
these experiments were not able to establish a definitive standard for      demonstrated in the figure: the growth curve of the Transformer-
determining the minimum sample requirements. On the contrary,               based model is more stable, but the curve of the CNNs model
the optimal dataset size may vary depending on specific applications

                            A                                           B

                            C                                           D

FIGURE 3

(A) Accuracy of the ViT model on small and medium-sized datasets: This plot demonstrates the training performance of the ViT model on datasets
with different sizes. (B) Accuracy of the ResNet101 model on small and medium-sized datasets: This plotdemonstrates the training performance of
the ResNet101 model on datasets with different sizes. (C) Comparison of the accuracy of ViT and ResNet101 for small and medium-sized datasets.
(D) Scatter plot comparison of model accuracy and FLOPs(floating-point operations per second) for the models.

Frontiers in Plant Science                                              07  frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                                                                                      10.3389/fpls.2023.1231903
                                   A                                       B

                                   C                                       D

FIGURE 4

Showing training accuracy and loss comparison between the proposed FOTCA model and compared models in the study. (A) Training accuracy
curve for all models. (B) Training loss curve for all models. (C) Partial training accuracy curve for all models. (D) Partial training loss curve for all
models.

fluctuates up and down consistently at 1% (in accuracy) and 0.03 (in           usually less jittery due to the relatively simple training process of the
loss), and DenseNet-169 even overfitted, with its accuracy and loss            self-attention layer. Meanwhile, the characteristics of the attention
at the 29th epoch values are the highest (99.8%) and lowest (0.072)            link are more suitable for training fine-grained plant leaf disease
of the whole iteration, respectively. Subsequently, the accuracy               image recognition tasks, which is further demonstrated inside the
gradually declined and finally stabilized at about 98.3%. This                 section4.4. At the same time, overfitting is sooner or later as long as
phenomenon may be due to thedifferent learning patterns and                    the network is large and deep enough, as indicated in (Hinton et al.
model depths of CNNs and Transformer-based models(containing                   (2012). The large depth of DenseNet-169 (model depth of 169)
FOTCA and ViT) during the training process.The convolutional                   reflects the overfitting of the model more quickly.
operation of CNN networks requires a large number of parameters.
In contrast, the ViT network uses aself-attention layer instead of a               Meanwhile, comparing the adjustment time of various models,
fixed convolutional layer, which is better able to cope with long              ViT's training results and FOTCA model performance are optimal,
sequences of image features. The training error of ViT networks is             both achieving near-maximum effects at the 9th epoch. Other CNN
                                                                               models are relatively slower and require higher computational costs

TABLE 2 Performance metrics of the proposed FOTCA model and compared models.

Model        Accuracy(%) Adjustment time F1 score(%)                           Loss   Adjustment time  Params(M)  FLOPS(G)

DenseNet169 98.3                      13                            91.85      0.082  14               14.15      3.44
                                                                               0.15   29               23.83      2.86
Inceptionv3  96.8                     28                            94.45      0.103  27               25.56      4.13
                                                                               0.107  32               44.55      7.87
ResNet50     99.5                     21                            98.23      0.206  30               78.14      29.96
                                                                               0.026  9                58.07      11.28
ResNet101    97.6                     32                            95.69      0.005  9                59.14      11.87

VGG-19       97.8                     28                            91.42

ViT          99.4                     9                             98.03

FOTCA        99.8                     11                            99.31

Model Accuracy(%)Adjustment F1-Loss Adjustment Params(M) FLOPS(G).

Frontiers in Plant Science                                                 08                                     frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                     10.3389/fpls.2023.1231903

to achieve the same training effect. Therefore, FOTCA model                   generates more errors and retains unnecessary information, thus
exhibits efficiency and accuracy in feature extraction, learning              reducing the recognition accuracy. At the same time, none of the
ability, and model convergence efficiency, making it a more                   CNNs can accurately and carefully focus on the correct plant leaf
excellent image recognition model.                                            disease feature points, and the fluctuating anomalies of the metrics
                                                                              (including accuracy and loss values) during the training process are
    Two categories of performance can be identified in the                    proved accordingly, and it also sideways proves the results that the
comparison experiments. One is represented by models such as                  training of Convolutional Networks is not as good as the ViT effect
VGG-19, DenseNet-169 and Inception-v3, whose accuracy did not                 in the recognition of plant leaf disease images.
reach 98% but has already approached the best performance, so the
trend is no longer rising. The other category is represented by                   In addition, during the recognition of maize grey leaf spot
models such as ViT, ResNet-50, and ResNet-101, which are based                leaves, the FOTCA and ViT models focused on the brown-black
on FOTCA proposed in this article. During the iteration process,              pockmarks on the leaves, and the ViT model focused more evenly
these models can approach an accuracy of 99.9%, which means that              on the background colour similar to the subject colour, Inception-
almost perfect image recognition results can be achieved through              v3 was able to observe the rectangular long stripes on the image
limited training. Additionally, FOTCA has leading positions in both           more accurately, while the rest of the models focused on looser areas
accuracy and F1-score indicators compared to other models, and it             and could not reflect a more convincing pattern. In turn, the rest of
is the only model with an F1-score of 0.9931. On the basis of                 the images show that the FOTCA model focuses more on the fine-
efficient recognition using ViT, FOTCA further improves accuracy,             grained features of the edges and less on the details of the whole leaf
loss, and F1-score without increasing parameter quantity and                  than the ViT model, and the five CNN models, on the other hand,
FLOPS compared to other models. This makes its performance                    have a more mixed focus, tending to focus on the global
nearly perfect, as shown in all the final parameters, parameter               information and main features of the images. From this we can
quantities, and FLOPS charts for all models in the experiment.                conclude that the Transformer-based models focuses more on the
Therefore, the FOTCA model proposed in this article achieves the              global features of the image, forming a global attention to the image,
best performance in plant leaf disease recognition.                           while the CNNs model focuses more on the subtle features, forming
                                                                              a local attention to the image.

4.4 Model visualization analysis                                              4.5 Ablation experiment

    We proceeded to discuss the differences between the models in                 To evaluate the effectiveness of each improvement module in
terms of their focus on image features by selecting 11 more                   the FOTCA model, we conducted ablation experiments using the
representative photographs from the dataset that contained                    original dataset. We compared four models: the FOTCA model, the
features that were essentially the basic features under the                   FOTCA model with original Pap Embedding, the FOTCA model
category. One photo of a healthy cherry leaf was taken with a                 with MLP-Classifier, and the FOTCA model lacking data
global angle of view of the whole leaf and with a portion of the              augmentation. We plotted the accuracy and loss curves for each
background included (this allows surveying the model's ability to             model as a function of epoch value in Figure 6. Detailed training
segment objects and backgrounds and verifying whether the model               data is presented in Table 3. Partial iteration accuracy and loss
can focus more on the objects themselves), with the diseased and              values are not shown here because the comparison of the individual
healthy traits shown on the leaf as curling at the edges of the               models in the results of this experiment can be shown more clearly
diseased leaf and wilting of the whole leaf, and one partial image of a       in Figure.
leaf with grey leaf spot A partial image of a maize leaf with grey leaf
spot, which does not include the background (this verifies the                4.5.1 Between FOTCA model and the model
model's ability to focus on fine-grained features of the object and           using original pap embedding
the size of the receptive field), and the trait of grey leaf spot on the
maize showing random grey-brown rectangular long or irregular                     Using Fourier transform can better capture texture and detail
long spots and brown-black pockmarks throughout the leaf, with                information in the image. This is because the Fourier transform can
additional longitudinal correlation images having similar features.           convert spatial-domain information into frequency-domain
The graph shows the attention of the seven models to the same                 information, allowing the model to better process this
image, where the warmer the colour, the more attention the model              information. In addition, using Fourier transform can greatly
pays to that feature, and the colour distribution gives an indication         reduce the dimensionality of the vector and thus reduce
of the model's ability to accurately identify the image features. Each        computation complexity.
model feature concern map is showing in the Figure 5.
                                                                                  However, Fourier transformation may cause some spatial
    First of all, both FOTCA and ViT models focus more accurately             information loss, thereby affecting the performance of image
on the details of cherry leaf edges, but in the results, it was found         recognition tasks. From the experimental results, it can be seen
that the ViT model focuses more on the whole leaf surface, which              that using Pap Embedding with Fourier operator can partially

Frontiers in Plant Science                                                09  frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                  10.3389/fpls.2023.1231903

FIGURE 5

Attention heatmaps for the models on representative plant leaf disease images. The attention heatmaps visually demonstrate the regions within the
image that the respective models focus on, highlighting their effectiveness at capturing relevant features for accurate recognition.

improve the accuracy and convergence speed of the model, thus              avoid gradient vanishing and information bottleneck problems, and
achieving better performance. Compared with using original Pap             has better explainability. In contrast, using MLP as the classifier will
Embedding, the accuracy was improved by 0.7% and the loss                  lose some important spatial structure information because MLP is a
decreased by 0.014.                                                        fully connected layer that flattens all features together and ignores
                                                                           their relationships. The classifier in this article can reconstruct spatial
4.5.2 Between FOTCA model and the model                                    structure information of images more accurately, retain more feature
using MLP - classifier                                                     information, and improve the model's recognition accuracy. At the
                                                                           same time, as a classifier, it can help reduce the risk of overfitting by
    The classifier used in this article contains shortcut connections      helping the model learn more robust features that have similar
across layers, which can help information transmit more effectively,

Frontiers in Plant Science                                             10  frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                                                        10.3389/fpls.2023.1231903

                            A                                               B

FIGURE 6

Showing training accuracy and loss comparison between the proposed FOTCA model and ablation experiment models with different parts removed.
(A) Training accuracy curve for all models. (B) Training loss curve for all models.

TABLE 3 Performance metrics of the proposed FOTCA model and ablation study models with different parts removed.

Model                             Accuracy(%)  Adjustment time                  F1-score(%)  loss                Adjustment time

FOTCA                             99.8         12                               99.31        0.005               9
Using the original Pap Embedding  99.1         12                               96.31        0.029               11
Using MLP-Classifier              96.1         12                               85.91        0.0341              15
Missing data augmentation         99.7         11                               98.94        0.019               12

responses for different input data, thus improving the model's                  5 Conclusions
generalization ability, which is also reflected in the training results.
                                                                                    Based on the above experimental results, the Transformer-based
4.5.3 Between FOTCA model and the model                                         models outperforms the CNNs in the field of plant leaf disease image
lacking data augmentation                                                       recognition, mainly because it allows a more detailed and accurate
                                                                                focus on image features., and can surpass the CNNs in terms of
    From the iteration graph, it can be observed that the model without         accuracy, loss, model matching speed, convergence efficiency, etc.
data augmentation behaves extremely similarly to the original model             The FOTCA model proposed in this paper can further improve its
during the process of iteration (the accuracy of the original model is          feature observation and extraction ability, and has a tendency to
99.8%, and the accuracy of the comparative model is 99.7%). Even in             improve in accuracy, loss and F1-score, while demonstrating the
the early stages of training, it produces better results than the original      enormous potential and application of adaptive Fourier operators.
model, possibly due to the large amount of pest and disease data in this        We will continue to extend our model in the future to combine
study, which allows for a sufficient sample size to collect fine-grained        diverse approaches for plant crop disease identification and detection
features without the need for data augmentation. However, continuous            in complex contexts in complex future contexts.
data augmentation during the initial stages of model training causes the
model to learn orientation features while learning the original features        Data availability statement
of the photos, increasing the computational load of the model,
resulting in a less effective performance on the test set compared to               The original contributions presented in the study are included
the model without data augmentation.                                            in the article/supplementary material. Further inquiries can be
                                                                                directed to the corresponding authors.
    For the model using the original Pap Embedding, its learning
ability is constrained for the same linkages, which adds limitations            Author contributions
to the model architecture. As for the model using MLP-Classifier, its
performance change is the largest among the four models,                            BH contributed to the conceptualization and design of the study
indicating significant contributions of the classifier used in this             and constructed improved experiments. WJ and JZ organized data
paper, which can significantly improve the fitting effect of the                organization and analysis validation. CC and LH wrote some
model. At the same time, the adjustment time of accuracy (loss)                 manuscripts. All authors contributed to manuscript revision, read,
for the four experiments is roughly the same, indicating that making            and approved the submitted version.
minor changes to the model architecture under the same model
does not have a significant effect on convergence efficiency. The
FOTCA model is still a fast-converging model that is leading in
plant leaf disease image recognition.

Frontiers in Plant Science                                                  11                                   frontiersin.org


--- PAGE BREAK ---

Hu et al.                                                                                                                    10.3389/fpls.2023.1231903

Funding                                                                                                Conflict of interest

    This article is supported by National Natural Science                                                  The authors declare that the research was conducted in the
Foundation of China (Grant/Award Number: "81460329") and                                               absence of any commercial or financial relationships that could be
the Natural Science Foundation of Jiangxi Province (Grant/Award                                        construed as a potential conflict of interest.
Numbers: "20192ACBL20039/202310454/202310456").

Acknowledgments                                                                                        Publisher's note

    We would like to express our sincere gratitude to all the                                              All claims expressed in this article are solely those of the authors
individuals who supported me throughout my research project.                                           and do not necessarily represent those of their affiliated organizations,
Their encouragement, moral support, and companionship were                                             or those of the publisher, the editors and the reviewers. Any product
invaluable and made this work possible.                                                                that may be evaluated in this article, or claim that may be made by its
                                                                                                       manufacturer, is not guaranteed or endorsed by the publisher.

References

  Berg, T., Liu, J. K., Woo Lee, S. A., Alexander, M. L. M., and Jacobs, D. W. (2014). "Birdsnap:        Lecun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning
Largescale fine-grained visual categorization of birds," in Proceedings of the IEEE/CVF                applied to document recognition. Proc. IEEE 86, 2278�2324. doi: 10.1109/5.726791
Conference on Computer Vision and Pattern Recognition (CVPR) (IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR)), IEEE, Columbus, OH, USA. 2011�2018.                     Lee, S. H., Lee, S., and Song, B. C. (2021). Vision transformer for small-size datasets.
                                                                                                       doi: 10.48550/arXiv.2112.13492
  Bisen, D. (2021). Deep convolutional neural network based plant species recognition
through features of leaf. Multimed. Tools Appl. 80, 6443�6456. doi: 10.1007/S11042-                      Lin, D., Shen, X., Lu, C., and Jia, J. (2015). "Deep lac: Deep localization, alignment
020-10038-W                                                                                            and classification for fine-grained recognition," in 2015 IEEE Conference on Computer
                                                                                                       Vision and Pattern Recognition (CVPR), Boston, MA, USA: IEEE. 1666�1674.
  Cai, C., Zhang, T., Weng, Z., and Feng, C. (2021). "A transformer architecture with                  doi: 10.1109/CVPR.2015.7298775
adaptive attention for fine-grained visual classification," in International Conference on
Computer and Communications (IEEE), Chengdu, China. 863�867.                                             Ouppaphan, P. (2017). "Corn disease identification from leaf images using
                                                                                                       convolutional neural networks," in Proceedings of the 2017 21st International
  Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S.                    Computer Science and Engineering Conference (ICSEC) (IEEE), Bangkok, Thailand.
(2020). "). End-to-end object detection with transformers," in Proceedings of the                      1�5. doi: 10.1109/ICSEC.2017.8443919
European Conference on Computer Vision, Springer, Glasgow, UK. 213�229, Springer.
                                                                                                         Pan, H., Xie, L., and Wang, Z. (2022). Plant and animal species recognition based on
  Chen, Y., Bai, Y., Zhang, W., and Mei, T. (2019). "Destruction and construction                      dynamic vision transformer architecture. Remote Sens. 14, 5242. doi: 10.3390/rs14205242
learning for fine-grained image recognition," in 2019 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) (IEEE), Long Beach, CA, USA.                              Sanju, S., and VelammaL, D. (2021). An automated detection and classification of
5152�5161. doi: 10.1109/CVPR.2019.00530                                                                plant diseases from the leaves using image processing and machine learning techniques:
                                                                                                       A state-of-the-art review. Ann. Rom. Soc Cell Biol. 25, 15933�15950.
  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
et al. (2020). "An image is worth 16x16 words: Transformers for image recognition at                     Sun, G. (2019). Fine-grained recognition: Accounting for subtle differences between
scale," in Proceedings of the International Conference on Learning Representations.                    similar classes. Proceedings of the AAAI conference on artificial intelligence. (New York,
https://OpenReview.net, virtual, Formerly Addis Ababa ETHIOPIA                                         USA: AAAI) 34 (07), 12047�12054. doi: 10.1609/aaai.v34i07.6882

  Fu, J., Zheng, H., and Mei, T. (2017). "Look closer to see better:recurrent attention                  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al.
convolutional ne ural network for fine-grained image recognition," in Proceedings of the               (2017). "Attention is all you need," in In Proceedings of the 31st International
IEEE conference on computer vision and pattern recognition, IEEE, Hawaii, USA. 4438�4446.              Conference on Neural Information Processing Systems (NIPS'17) (Curran Associates
                                                                                                       Inc.), Long Beach California USA. 6000�6010.
  Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., and Catanzaro, B. (2021).
Adaptive fourier neural operators: Efficient token mixers for transformers. arXiv                        Wang, Y., Huang, R., Song, S., Huang, Z., and Huang, G. (2021). "Not all images are
preprint. arXiv:2111.13587. doi: 10.4505/arXiv.2111.13587                                              worth 16x16 words: Dynamic transformers for efficient image recognition," in
                                                                                                       Proceedings of the Neural Information Processing Systems. NeurIPS, virtual
  He, J., Chen, J.-N., and Liu, S. (2022). "Transfg: A transformer architecture for fine-
grained recognition," in Proceedings of the AAAI Conference on Artificial Intelligence,                  Wang, D., Wang, J., Li, W., and Guan, P. (2021). T-cnn: Trilinear convolutional
AAAI, Pomona, California. 852�860.                                                                     neural networks model for visual detection of plant diseases. Comput. Electron. Agric.
                                                                                                       190, 106468. doi: 10.1016/j.compag.2021.106468
  Hinton, G. E., Srivastava, N., Krizhevsky, A., Krizhevsky, A., Sutskever, I., and
Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of                    Wei, M., Wu, Q., Ji, H., Wang, J., Lyu, T., Liu, J., et al. (2023). A skin disease
feature detectors. arXiv preprint arXiv:1207.0580. doi: 10.9774/GLEAF.978-1-909493-38-42               classification model based on densenet and convnext fusion. Electronics 12, 438.
                                                                                                       doi: 10.3390/electronics12020438
  Iqbal, Z., Sharif, M., Shah, J., ur Rehman, M., and Javed, K. (2018). An automated
detection and classification of citrus plant diseases using image processing techniques:                 Yang, Z., Luo, T., Wang, D., Hu, Z., Gao, J., and Wang, L. (2018).
A review. Comput. Electron. Agric. 153, 12�32. doi: 10.1016/j.compag.2018.07.041                       "Learning to navigate for fine-grained classification," in Computer vision � ECCV
                                                                                                       2018, vol. 11218 . Ed. V. Ferrari (Munich, Germany: Springer). H. M. S. C. W. Y.
  Kong, J., Wang, H., Wang, X., Jin, X., Fang, X., and Lin, S. (2021). Multi-stream                    doi: 10.1007/978-3-030-01264-9\26
hybrid architecture based on cross-level fusion strategy for finegrained crop species
recognition in precision agriculture. Comput. Electron. Agric. 185, 106134. doi: 10.1016/                Zhang, N., Donahue, J., Girshick, R. B., and Darrell, T. (2014). Part-based R-CNNs
j.compag.2021.106134                                                                                   for fine-grained category detection. Computer Vision�ECCV 2014: 13th European
                                                                                                       Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13.
  Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). Imagenet classification with                (Zurich, Switzerland: Springer), 834�849. doi: 10.1007/978-3-319-10590-1_54
deep convolutional neural networks. Commun. ACM 60, 84�90. doi: 10.1145/3065386
                                                                                                         Zhang, S., and Wang, Z. (2016). Cucumber disease recognition based on global-local
  Kumar, S., Gupta, S. K., Kaur, M., and Gupta, U. (2023a). Vi-net: A hybrid deep                      singular value decomposition. Neurocomputing 205, 341�348. doi: 10.1016/
convolutional neural network using vgg and inception v3 model for copy-move forgery                    j.neucom.2016.04.034
classification. J. Vis. Comun. Image Represent. 89, 103644. doi: 10.1016/j.jvcir.2022.103644
                                                                                                         Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment
  Kumar, S., Pal, S., Singh, V. P., and Jaiswal, P. (2023b). Performance evaluation of                 using multitask cascaded convolutional networks. IEEE Signal Process. Lett. 23, 1499�
resnet model for classification of tomato plant disease. Epidemiologic Methods 12,                     1503. doi: 10.1109/LSP.2016.2603342
20210044. doi: 10.1515/em-2021-0044

Frontiers in Plant Science                                                                         12                        frontiersin.org


--- PAGE BREAK ---



---

## EXTRACTED FIGURES (visual analysis)

### Figure 2 - FOTCA model architecture and two fine-tuning losses (page 4, `papers/21_AFNO/inputs/figures/page4.png`)
(A) Model: input image 224 px x 224 px (stack of images) -> split into patches -> "Patch embedding and flatten patches" block: FFT -> Conv2d -> IFFT. Output is added (circled +) with "Position encoding based on learnable Fourier features". Result goes to "Adaptive Fourier Neural Operators" block, repeated "L Layers": Spatial Mixing -> Channel Mixing (down arrow between them). Zoomed-in side detail of the Fourier block: FFT -> Multiply Frequency -> IFFT. Then "Class token" -> "Classifier". No kernel sizes/strides/channel numbers are drawn in the figure.
(B) Cross-entropy loss for fine-tuning: X(k) -> FOTCA model -> X(k+1) -> FOTCA model -> X(k+2); L2 terms on X_ref(k+1), X_ref(k+2), summed to loss.
(C) Focal loss: same flow, with L2 terms weighted by -(1-p_t)^gamma.
Note: the figure is a conceptual flowchart; hidden size/depth etc. come only from Table 1 (below).

### Table 1 - Hyperparameters (page 6, `figures/page6.png`) - verified visually
- Initial learning rate 0.001
- Epochs 100
- Batch size 8
- Image size (all) 224*224 ; Inception-v3 299*299
- Transformer-based models: Embedding dim 768, Patch size 16, Head number 8, Depth 12

### Table 2 - comparison (page 8, `figures/page8.png`) - verified visually
Model | Acc% | AdjTime(acc) | F1% | Loss | AdjTime(loss) | Params(M) | FLOPS(G)
DenseNet169 | 98.3 | 13 | 91.85 | 0.082 | 14 | 14.15 | 3.44
Inceptionv3 | 96.8 | 28 | 94.45 | 0.15 | 29 | 23.83 | 2.86
ResNet50 | 99.5 | 21 | 98.23 | 0.103 | 27 | 25.56 | 4.13
ResNet101 | 97.6 | 32 | 95.69 | 0.107 | 32 | 44.55 | 7.87
VGG-19 | 97.8 | 28 | 91.42 | 0.206 | 30 | 78.14 | 29.96
ViT | 99.4 | 9 | 98.03 | 0.026 | 9 | 58.07 | 11.28
FOTCA | 99.8 | 11 | 99.31 | 0.005 | 9 | 59.14 | 11.87
(Figure 4 on page 8: validation accuracy/loss curves over 100 epochs for all 7 models; FOTCA near 1.0 acc and ~0.005 loss from very early epochs.)

### Table 3 - ablation (page 11, `figures/page11.png`) - verified visually
Model | Acc% | AdjTime | F1% | loss | AdjTime
FOTCA | 99.8 | 12 | 99.31 | 0.005 | 9
Using the original Pap Embedding | 99.1 | 12 | 96.31 | 0.029 | 11
Using MLP-Classifier | 96.1 | 12 | 85.91 | 0.0341 | 15
Missing data augmentation | 99.7 | 11 | 98.94 | 0.019 | 12
(Figure 6: ablation validation curves; MLP-classifier variant is slowest and plateaus ~0.96.)

---

## TRAINING SETUP SUMMARY (from text, sections 3.1, 3.2, 4.1, 4.2)
- Framework: PyTorch 1.11.0, Python 3.8, CUDA 11.3, single V100-32GB.
- Dataset: PlantVillage, 54,303 images, 38 classes (13 plant species); the text says non-plant-leaf images were inserted. Train:val = 8:2 (paper's own split; no separate test set mentioned - validation curves are the reported results).
- Image size 224x224 (Inception-v3 299x299).
- Augmentation (train only): RandomFlip, RandomCrop, RandomHorizontalFlip, RandomResizeCrop (exact params not given).
- Optimizer: SGD (momentum/weight decay not stated); LR 0.001 initial, scheduler LambdaLR with linear decay by number of LR updates (exact lambda not given).
- Batch 8, epochs 100, dropout used (rate not stated).
- Loss: focal loss for FOTCA (gamma not stated), cross-entropy for all other compared models.
- Transfer learning: all models initialised from ImageNet pretrained weights; FOTCA = ViT-like backbone (D=768, patch 16, heads 8, depth 12) with AFNO token mixing, learnable-Fourier-feature positional encoding, and a CNN shortcut-block classifier (two 3x3 conv + ReLU, 1x1 conv shortcut with ReLU, residual add then ReLU; BatchNorm mentioned in 3.2.3; widths not given).
- Metrics: accuracy, loss, F1, params, FLOPs, "adjustment time" (iterations/epochs to reach 95% of final converged performance). Baselines: DenseNet-169, Inception-v3, VGG19, ResNet-50, ResNet-101, ViT.
- Missing info (do not guess): momentum, weight decay, gamma, dropout rate, number of AFNO basis functions K, MLP ratio, classifier channel widths, how pretrained weights map onto FOTCA.
