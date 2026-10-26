# PAPER CONTENT: 16_fpls

**Source PDF:** papers/16_fpls/inputs/fpls-14-1328952.pdf
**Extracted:** 2026-10-20

---

                                                    TYPE Original Research
                                                    PUBLISHED 12 January 2024
                                                    DOI 10.3389/fpls.2023.1328952

OPEN ACCESS                                     Automatic classiﬁcation of
                                                ligneous leaf diseases via
EDITED BY                                       hierarchical vision transformer
                                                and transfer learning
Jian Lian,
Shandong Management University, China           Dianyuan Han and Chunhua Guo*

REVIEWED BY                                     Media and Communications College of Weifang University, Weifang, Shandong, China

Wei Meng,                                       Background: Identiﬁcation of leaf diseases plays an important role in the growing
Beijing Forestry University, China              process of different types of plants. Current studies focusing on the detection and
Yuwan Gu,                                       categorization of leaf diseases have achieved promising outcomes. However, there
Changzhou University, China                     is still a need to enhance the performance of leaf disease categorization for
Xiaoyang Liu,                                   practical applications within the ﬁeld of Precision Agriculture.
Huaiyin Institute of Technology, China
                                                Methods: To bridge this gap, this study presents a novel approach to classifying
*CORRESPONDENCE                                 leaf diseases in ligneous plants by offering an improved vision transformer model.
                                                The proposed approach involves utilizing a multi-head attention module to
Chunhua Guo                                     effectively capture contextual information about the images and their classes.
    gch58@163.com                               In addition, the multi-layer perceptron module has also been employed. To train
                                                the proposed deep model, a public dataset of leaf disease is exploited, which
RECEIVED 27 October 2023                        consists of 22 distinct kinds of images depicting ligneous leaf diseases.
ACCEPTED 15 December 2023                       Furthermore, the strategy of transfer learning is employed to decrease the
PUBLISHED 12 January 2024                       training duration of the proposed model.

CITATION                                        Results: The experimental ﬁndings indicate that the presented approach for
                                                classifying ligneous leaf diseases can achieve an accuracy of 85.0% above.
Han D and Guo C (2024) Automatic
classiﬁcation of ligneous leaf diseases         Discussion: In summary, the proposed methodology has the potential to serve as
via hierarchical vision transformer             a beneﬁcial algorithm for automated detection of leaf diseases in ligneous plants.
and transfer learning.
Front. Plant Sci. 14:1328952.
doi: 10.3389/fpls.2023.1328952

COPYRIGHT

© 2024 Han and Guo. This is an open-access
article distributed under the terms of the
Creative Commons Attribution License (CC BY).
The use, distribution or reproduction in other
forums is permitted, provided the original
author(s) and the copyright owner(s) are
credited and that the original publication in
this journal is cited, in accordance with
accepted academic practice. No use,
distribution or reproduction is permitted
which does not comply with these terms.

                                                KEYWORDS

                                                precision agriculture, transformer, neural networks, machine vision, transfer learning

                                                1 Introduction

                                                    The occurrence of leaf diseases in plants holds signiﬁcant relevance in the ﬁeld of plant
                                                pathology. Severe leaf disease can have detrimental effects on plants, including leaf drying
                                                and hindered bud formation. It can weaken the health of the plant and worsen the
                                                susceptibility to other diseases Kai et al. (2011); Bo et al. (2019); Xu et al. (2020); Wang et al.
                                                (2021; 2022). In addition, the occurrence of fruit leaf disease can lead to a decline in both

Frontiers in Plant Science                      01  frontiersin.org
Han and Guo                                                                    10.3389/fpls.2023.1328952

the quantity and quality of fruits, as well as increase the vulnerability      Finally, to implement the classiﬁcation task, they employed a CNN
of nearby plants to infection. Given the strong reliance of the                model. To effectively identify olive leaf disease, Ksibi et al. (2022)
economy on agricultural productivity, the impact of the leaf disease           proposed the utilization of ResNet50 and MobileNet models for
on the environment becomes particularly signiﬁcant if preventive               image feature extraction, employing the technique of feature
measures are not implemented in a timely manner. Therefore, the                concatenation. To train the deep learning models employed in this
prompt identiﬁcation of diseases affecting fruit leaves is crucial for         investigation, a dataset including 5,400 images of olive leaves was
human well-being Patil et al. (2017); Afzaal et al. (2021); Mahum              utilized. These images were acquired from an olive grove using an
et al. (2022). In general, the identiﬁcation and categorization of leaf        unmanned aerial vehicle (UAV) equipped with a camera. In their
diseases have predominantly depended on the human visual system,               study, Devi et al. (2022) proposed a methodology for the prediction
which is error prone, is time-consuming and labor-intensive.                   and classiﬁcation of corn leaf disease. The authors employed transfer
Hence, the implementation of automated leaf disease                            learning and the Alexnet model, leveraging the Adaptive Moment
classiﬁcation is imperative in the context of fruit production for             Estimation (ADAM) optimizer and the Stochastic Gradient Descent
mitigating both the production and economic losses Sneha and                   with momentum (SGDM) mechanism. The model was trained and
Bagal (2019); JencyRubia and BabithaLincy (2021); Y et al. (2022).             evaluated using a dataset consisting of 5,300 images, which were
                                                                               categorized into four different types: healthy, blight, common rust,
    In recent decades, there has been a signiﬁcant surge in the                and gray leaf spot. Yao et al. (2022) conducted a study focusing on the
utilization of machine learning-based algorithms for addressing leaf           identiﬁcation of kiwifruit leaf disease. They developed a publicly
disease categorization problems. Numerous machine vision                       available dataset while using the YOLOX target detection algorithm
algorithms have been proposed to classify illnesses affecting plant            to mitigate the inﬂuence of environmental elements. The study of Yu
leaves. In the study conducted by Singh and Misra (2017), the                  et al. (2022) introduced a method for efﬁciently detecting soybean
authors proposed an image segmentation method for the automatic                illnesses. It leverages a residual attention network (RANet) model.
identiﬁcation and classiﬁcation of plant leaf diseases, speciﬁcally            This study included the incorporation of three types of soybean leaf
focusing on the minor leaf disease common in pine trees within the             spot diseases, namely soybean brown leaf spot, soybean frog eye leaf
United States. The researchers investigated the utilization of several         spot, and soybean phyllosticta leaf spot, into the dataset. The OTSU
classiﬁer algorithms for the purpose of identifying plant leaf disease.        algorithm was utilized to pre-process the initial images for
A system for automatic detection of plant disease using image                  eliminating the surrounding features. Additionally, the image
processing techniques was proposed by the authors Mounika and                  dataset was augmented by the application of image enhancement
Bharathi (2020). The approach was used for calculation of textural             algorithms. Additionally, the residual attention layer was constructed
data pertaining to illnesses affecting plant leaves. In their work,            by integrating attention processes into a ResNet18 model.
Kulkarni and Sapariya (2021) proposed a method to automatically
detect and classify leaf illnesses, which encompasses many stages,                 The majority of the preceding approaches in the ﬁeld of leaf
including image gathering, image pre-processing, segmentation,                 disease classiﬁcation have predominantly employed CNN
and classiﬁcation. In their study, Reddy et al. (2021) employed                architectures. Regrettably, the CNN-based models have
Support Vector Machine (SVM) and Random Forest algorithms for                  limitations due to the local receptive ﬁeld inside the convolutional
the purpose of detecting illnesses in leaves. This study compared              modules. This characteristic directs attention towards the
assessment measures, such as Root Mean Square Error (RMSE),                    surrounding region in an image, perhaps overlooking the
Peak Signal Noise Ratio (PSNR), for the diseaseaffected regions of             connections between distant pixels. In contrast, the transformer is
the leaves to assess their potential impact on agricultural output.            renowned for its utilization of an attention mechanism to effectively
                                                                               capture and represent the extensive inter-dependencies within the
    In recent years, deep learning has gained signiﬁcant interest due          data samples. The successful performance of transformer in NLP
to its remarkable achievements in many domains, such as natural                tasks has resulted in its integration and use in the ﬁeld of computer
language processing (NLP) and machine vision. Consequently, there              vision Liu et al. (2021). For instance, the work conducted by Qian
have been additional advancements in the ﬁeld of plant leaf disease            et al. (2022) introduced a novel strategy for classifying maize leaf
categorization by the utilization of deep learning models. Liu et al.          diseases using a vision transformer-based method. The authors of
(2017) introduced a methodology for detecting apple leaf diseases              the study also gathered RGB images from publicly available
utilizing deep convolutional neural networks (CNNs). The model                 databases and experimental ﬁelds, classifying them into four
reported in this study is capable of generating an ample number of             distinct categories: southern corn leaf blight, gray leaf spot,
diseased images with a deep learning model, AlexNet. The study                 southern corn rust, and healthy specimens. Nevertheless, the
utilized a dataset including 13,689 images depicting various apple leaf        vision transformer model proposed in this study might provide
illnesses. The CNN model developed in this research was trained to             challenges when used to high-resolution images due to the
accurately classify four types of apple leaf diseases. In the study            quadratic computational complexity of the self-attention
conducted by Anagnostis et al. (2020), a resilient CNN model was               mechanism in relation to image resolution. Furthermore, the
developed to address the timely identiﬁcation of anthracnose, a                original vision transformer necessitates a substantial allocation of
prevalent fungal disease that affects numerous tree species globally.          memory capacity and processing resources.
This model was to use to classify images of plant leaves as either
infected or uninfected by anthracnose. The researchers acquired a                  Taking the aforementioned research into consideration, we
dataset consisting of grayscale and RGB images. Then, they utilized a          propose a hierarchical vision transformerbased approach by
rapid Fourier transform to extract characteristics from the images.            employing transfer learning strategy, for classifying leaf diseases of

Frontiers in Plant Science                                                 02  frontiersin.org
Han and Guo                                                                                                        10.3389/fpls.2023.1328952

ligneous plants. The hierarchical design in the proposed vision               inspiration from the work of Chen et al. (2020). To achieve this,
transformer yields notable reductions in computational resource               the proposed model is initially trained on the ImageNet dataset
requirements and the number of weighting parameters for the                   Russakovsky et al. (2014), considering the relatively small size of the
vision transformer. Furthermore, this work utilizes the weighting             presented image dataset. In addition, the images are resized into a
factors that were pre-trained on the dataset ImageNet Russakovsky             uniform dimension of 224×224 to minimize the computing
et al. (2014). To assess the effectiveness of the suggested methodology,      resources needed during the training phase. Moreover, the
a subset of a publicly available dataset was utilized. This subset            present study employs a set of data augmentation techniques to
comprises a total of 22 types of ligneous leaf images. Furthermore, a         increase the number of image samples, which can further enhance
series of comparative tests were carried out to evaluate the                  the generalization of the proposed model and mitigate the risk of
performance of the suggested methodology as well as the state-of-             over-ﬁtting during the training process. These techniques include
the-art methods. The experimental ﬁndings provide evidence that the           RandomFlip, Color Jitter, Cutmix Yun et al. (2019), and Mixup
suggested methodology outperforms the state-of-the-art techniques             Zhang et al. (2017).
in terms of accuracy, precision, recall, and, F1 score.
                                                                              2.2 Overall framework
    In general, the contributions of this study include:
                                                                              The proposed vision transformer model is provided in Figure 2,
     • A leaf disease classiﬁcation pipeline is proposed. The
         proposed model primarily consists of a hierarchical                  which is a typical two-channel swin vision transformer Liu et al.
         vision transformer.
                                                                              (2021) model, and there is no weighting parameter sharing between
     • The presented vision transformer model comprises of two
         channels, which are used to extract the features from the            these two channels.
         original leaf images and the edges in the corresponding
         images, respectively.                                                As seen in Figure 3, the input of the lower channel is achieved

     • The experimental ﬁndings prove the superiority of the                  by the utilization of the Sobel operator Liu and Wang (2022) and
         proposed methodology over the state-of-the-art algorithms.
                                                                              the continuous image fusion operation. The edge Sobel operator is
    The subsequent sections of this article are structured in the
following manner. Section 2 presents an elaborate exposition of the           employed on the original image in order to provide input for the
suggested transformer concept. Section 3 provides a detailed
account of the experimental methodology employed in this study,               suggested methodology. Initially, the gray-scale equivalent is
as well as the subsequent ﬁndings and their analysis. Finally, The
study concludes at Section 4.                                                 derived from each original image. Next, the original image

2 Methodology                                                                 undergoes convolution with the Sobel operators of size 3×3 in
                                                                              both the horizontal and vertical axes. The speciﬁc characteristics of
2.1 Dataset collection and pre-processing
                                                                              the horizontal and vertical Sobel operators, denoted as Gx and Gy
    The dataset utilized in this research is sourced from the publicly        respectively, are outlined below in Equations 1 and 2.
accessible plant dataset of AI Challenger 2018 Wu et al. (2017),
which has a total of 10 plant specimens, each classiﬁed into one of                                2  +1 0 −1  3
27 categories representing either leaf diseases or healthy conditions.
In a systematic manner, a total of 61 image classes have been                                      Gx = 664 +2 0 −2 757 Â I,  (1)
categorized into distinct groups based on species, pest species, and
severity levels. The objective of this work is to categorize diseases                                 +1 0 −1
affecting ligneous fruit leaves. Therefore, only the leaves that were
affected by diseases were selected from the dataset for the purposes                               2  +1 +2 +1  3
of training and validation. In this study, a total of 22 categories of
images depicting leaf diseases were included in the dataset. These                                 Gy = 664 0 0 0 775 Â I,    (2)
categories encompassed both sick leaves and healthy leaves.
                                                                                                      −1 −2 −1
    As seen in Figure 1, the training set comprises 11,603 images,
whereas the testing set consists of 1,668 images. These images are                where the original image is taken as I, and let Gx be equal to the
categorized into 22 distinct classes. Furthermore, the dataset                transpose of Gy. It is worth noting that the elements in the operators
includes a collection of example images, as seen in Figure 1.                 Gx and Gy are differentiable. The starting values of the convolutional
These images encompass both healthy and sick leaves.                          layer, also known as the Sobel operator layer, are determined by the
                                                                              elements in the Gx and Gy operators. These values may be optimized
    In this study, the utilization of transfer learning is employed to        by a back-propagation approach during the training phase of the
improve the performance of the proposed approach, taking                      proposed transformer. In addition to combining the output of these
                                                                              two channels through concatenation, the classiﬁcation process
                                                                              involves the utilization of a softmax classiﬁer, an average pooling
                                                                              layer, and a fully-connected layer.

                                                                              2.2.1 Details of the backbone

                                                                                  As seen in Figure 2, the conﬁguration of blocks in each channel
                                                                              and the size of tokens may be adjusted to accommodate diverse
                                                                              scales of machine vision applications. In accordance with the
                                                                              present investigation, the quantity of blocks in each channel is

Frontiers in Plant Science                                                03                                                  frontiersin.org
Han and Guo                                                                10.3389/fpls.2023.1328952

FIGURE 1

A collection of sample images depicting various types of leaf diseases. The leaves in the top row exhibit signs of good health. The leaves exhibiting
signs of illness are seen in the bottom row.

multiplied by a factor of 2, 2, 6, and 2, respectively. Following the      feature representation of a token is obtained by concatenating the
input technique, the input image is initially partitioned into non-        pixel values inside the associated patch. Different from the original
overlapping patches of size 4×4. Hence, the feature dimension of a         vision transformer, the proposed transformer model leverages the
single patch may be calculated as the product of its width, height,        swin transformer block and the shift-window self-
and number of color channels, resulting in a value of 48 (where 3          attention mechanism.
represents the number of RGB channels). In a manner akin to the
vision transformer proposed by Dosovitskiy et al. (2020), the                  In the initial stage, a linear embedding layer is employed to
approach involves treating each patch as a token, where the                project the original feature into a dimension of arbitrary size (C=96
                                                                           in the context of this work). Next, a series of swin transformer

FIGURE 2

The suggested model consists of a two-channel swin vision transformer, which exhibits a certain overall structure. The top channel of the proposed
model receives an initial image as its input, while the lower channel gets the edge information of the original image as its input. It is worth noting
that the value of C, which is equal to 96, might vary depending on the architecture of the model.

Frontiers in Plant Science                                             04  frontiersin.org
Han and Guo                                                                                      10.3389/fpls.2023.1328952

    FIGURE 3

    The formation of the input for the bottom channel of the proposed vision transformer model.

blocks are utilized on the tokens, incorporating two distinct forms                              z^l+1 = SW − MSA(LN(zl)) + zl,  (5)

of self-attention modules. Furthermore, it should be noted that the

number of tokens in the swin transformer blocks stays consistent                                 zl+1 = MLP(LN(z^l+1)) + z^l+1,  (6)

with the linear      embedding     unit,  which is   calculated  as  H   Â  W   .
                                                                     4      4
                                                                                                where the notation W-MSA refers to window-based MSA, MLP
    The hierarchical representation is generated by the provided                            stands for multiple layer perception Tolstikhin et al. (2021), SW-
                                                                                            MSA represents shifted-window MSA, and LN signiﬁes layer
model through the utilization of patch merging modules, which                               normalization Ba et al. (2016).

effectively down-sample the feature resolutions by a factor of 2. The

ﬁrst merger module and feature modiﬁcation are denoted as Stage 2,

which are then repeated as Stage 3 and Stage 4. Furthermore, the

dimensions of the output features progress from Stage 1 to Stage 4                          2.2.3 Shifted window-based self-
                                                                                            attention mechanism
as     H   Â   H  Â  C,   H  Â  H  Â  C,  H   Â  H   Â  C,  and      H   Â  H   Â  C,
       4       4          8     8         16     16                  32     32                  In contrast to the initial vision transformer that heavily relies on
                                                                                            global self-attention, which necessitates calculating the relationships
respectively. The hierarchical representation is primarily                                  between a token and all other tokens, the window-based MSA
                                                                                            module employs a window of size M × M (with a default value of
distinguished between the swin vision transformer Liu et al.                                M=7) to restrict the extent of calculation. Hence, the computational
                                                                                            complexity becomes more manageable with the incorporation of
(2021) and the original vision transformer Dosovitskiy et al.                               the window-based self-attention mechanism, as opposed to the
                                                                                            quadratic complexity of the vision transformer Dosovitskiy et al.
(2020) by the inclusion of Stage 2, Stage 3, and Stage 3 together.                          (2020), which is dependent on the image resolution h × w (as shown
                                                                                            in Equations 7, 8).
The given methodology does not include the utilization of any class

taken. In this approach, the output vector of dimensions N =

H   Â  W   is  generated  by  using   global  average       pooling  followed   by  a
32     32

fully-connected layer. The linear classiﬁer then takes into account

the ﬁrst C components of this output vector.

2.2.2 Swin transformer block                                                                     Ω(MSA) = 4hwC2 + 2(hw)2C,       (7)

    Each stage of the proposed model consists of the swin                                        Ω(W − MSA) = 4hwC2 + 2M2hwC,    (8)
transformer blocks. And each swin transformer block consists of
consecutive modules, as shown in Figure 4. In this architecture,                                where h and w denote the height and width of an image, C=96,
there are two important modules W-MSA and SW-MSA, which                                     and M=7 in the following settings.
represent the multi-head self-attention (MSA) with a standard
window and the MSA with a shifted window, respectively.                                         Furthermore, the SW-MSA strategy is intended to enhance the
                                                                                            encoding of global relationships among the pixels in multiple windows.
    The mathematical representation of the consecutive swing                                The use of the relationship across many windows may be maximized
transformer modules can be articulated in Equations 3–6:                                    with the introduction of SW-MSA. As seen in Figure 5, the partitioning
                                                                                            method of the regular window is employed in layer l, where self-
                         zbl = W − MSA(LN(zl−1)) + zl−1,                           (3)      attention is computed within each window. In the subsequent layer,
                                                                                            denoted as l + 1, the partitioning of the window is adjusted both
                             zl = MLP(LN(zbl)) + zb,                               (4)

Frontiers in Plant Science                                                              05                                       frontiersin.org
Han and Guo                                                                     10.3389/fpls.2023.1328952

FIGURE 4

The detailed components inside the Swin Transformer model. The abbreviation LN is used to refer to layer normalization. The normal and shifted-windows
multi-head self-attention modules are denoted as W-MSA and SW-MSA, respectively. The acronym MLP stands for multiple-layer perception. .

horizontally and vertically, resulting in the creation of a greater number      and the Python programming language version 3.8.3. The backbone of
of distinct windows. Thus, the self-attention calculation in Layer l+1          the suggested model consists of the Swin-T vision transformer, which is
traverses the initial windows in Layer l.                                       employed for each channel. The dimensions of the input images are
                                                                                standardized to 224×224. Furthermore, the suggested swin vision
    It should be noted that the loss function employed in the                   transformer was initialized using the pre-trained weighting
proposed model is a cross-entropy loss. This loss is computed by                parameters of ImageNet Russakovsky et al. (2014). Typically, the
comparing the ground truth category of the image with the                       hyper-parameters employed in the experiments encompass the
classiﬁcation output given by the suggested model, as seen in Figure 2.         subsequent elements, as shown in Table 1 To note that the
                                                                                experiments by using the proposed approach were conducted in a
3 Experiments                                                                   10-fold cross-validation scheme. Meanwhile, the hyper-parameters
                                                                                were determined by using a trial-and-error strategy.
3.1 Implementation details
                                                                                    In order to assess the effectiveness of the suggested model and
    The tests were done employing four NVIDIA RTX 3080 GPUs, the                the comparison methodologies, the experiments contained several
PyTorch deep learning framework Paszke et al. (2019) version 2.0.1,             assessment measures, including accuracy, precision, recall, and F1
                                                                                score (as shown in Equations 9–12).

FIGURE 5

The diagram depicting the SW-MSA mechanism employed in the proposed methodology. The red boxes are used to indicate the local window,
which serves the purpose of constraining the scope of self-attention calculation.

Frontiers in Plant Science                                                  06  frontiersin.org
Han and Guo                                                                                                              10.3389/fpls.2023.1328952

             Accuracy       =  (TP     (TP + TN)   FN )  ,  (9)                model under consideration has demonstrated a performance
                                      + TN + FP +                              improvement of 2.22% and 1.40% compared to the vision
                                                                               transformer version and the ﬁxed Sobel operator version,
                            Precision = TP ,                (10)               respectively, when evaluated on 50% of the identical dataset.
                                      TP + FP                                  Hence, the selected model was deemed suitable as the foundational
                                                                               framework for the subsequent investigations.
                            Recall =  TP       ,            (11)
                                                                               3.3 Experimental results
                                      TP + FN
                                                                                   To evaluate the performance of the proposed approach in a fair
             F1 = 2 Â Precision Â Recall                    (12)               manner, the comparison experiments were conducted between the
                            Precision + Recall,                                state-of-the-art methods, including, and ours on the same dataset as
                                                                               provided in Table 2.
    where TP, TN, FP, and FN denote number of true positive, true
negative, false positive, and false negative, respectively.                        In order to objectively assess the performance of the proposed
                                                                               approach, a series of comparative experiments were conducted.
3.2 Ablation study                                                             These experiments involved benchmarking the proposed approach
                                                                               against several state-of-the-art methods, namely AlexNet
    The proposed vision transformer model incorporates two                     Krizhevsky et al. (2012), GoogleNet Szegedy et al. (2014), VGG
distinct topologies for swin vision transformers. To evaluate the              Abas et al. (2018), ResNet101 Zhang (2021), EfﬁcientNetB3 Singh
efﬁcacy of the introduced swin vision transformer, a series of                 et al. (2022), Inception V3 Jenipher and Radhika (2022), MobileNet
ablation experiments were conducted on a publicly available                    V2140 Elfatimi et al. (2022), and vision transformer Dosovitskiy
dataset. These experiments involved varying the settings of the                et al. (2020). In the experiments, these state-of-the-art methods
introduced models, which were used to replace the original settings            adopted their original settings in the literature. To note that the
of the proposed approach. The original approach consisted of a                 former seven state-of-the-art algorithms are CNN models. And the
vision transformer Dosovitskiy et al. (2020) and the Sobel operator            proposed approach was inspired by the work of the last model
with ﬁxed 3×3 values.                                                          vision transformer. Meanwhile, the evaluation was carried out on
                                                                               the dataset speciﬁed in Table 2.
    As seen in Figure 6, it is evident that the accuracy of the suggested
methodology surpasses that of the model utilizing the original vision              As seen in Table 3, the suggested strategy exhibits superior
transformer or the ﬁxed Sobel operator. The transformer model                  accuracy, precision, recall, and F1 score compared to existing state-
under consideration has demonstrated a performance improvement                 of-the-art approaches. To provide speciﬁc results, our method
of 2.2% and 1.4% compared to the vision transformer version and the            demonstrates an increase in overall accuracy of 2.1% when
ﬁxed Sobel operator version, respectively, when evaluated on a subset          compared to MobileNet V2140. Additionally, our proposed
comprising 25% of the utilized dataset. Furthermore, the transformer           approach exhibits improvements in Precision, Recall, and F1 score

FIGURE 6

Ablation study with different settings with ratios (25% and 50%) of the training set in the publicly available dataset.

Frontiers in Plant Science                                                 07                                            frontiersin.org
Han and Guo                                                                                                  10.3389/fpls.2023.1328952

TABLE 1 Hyper-parameters used in the experiments.                            TABLE 2 Distribution of the images in the dataset of this study.

Item                                               Value                     Class Label Name        No. of train-  No. of
                                                                                                      ing images    testing
Batch_size                                                 8                                                        images
optimizer                                            Adam                     1 Apple healthy        1,185
learning rate                                                                                         211               169
depth                                                  1e-4                   2 Apple_Scab general    152               30
epochs                                                    12                                          427               22
                                                        100                   3 Apple_Scab serious    142               61
                                                                                                       40               20
by 2.6%, 2.7%, and 2.7% respectively, when compared to MobileNet              4 Apple Frogeye Spot    598
V2140. Furthermore, even when compared to the original vision                                         116                6
transformer, our approach showcases enhancements in accuracy,                 5 Cedar Apple           110
Precision, Recall, and F1 score by 1.1%, 1.5%, 0.59%, and 1.1%                      Rust general      294               85
respectively. In summary, the suggested methodology demonstrates                                      381               12
higher performance compared to both CNN-based and vision                      6 Cedar Apple           462
transformer-based algorithms. This provides evidence of the                         Rust serious      503               18
prospective capability of the proposed technique in feature extraction.                               419
                                                                              7 Cherry healthy         61               42
    In order to assess the effectiveness of the suggested methodology                                 630               54
on various image categories within the leveraging dataset, we have            8 Cherry_Powdery        367
included the accuracy-based confusion matrix (as seen in Figure 7)                  Mildew general   1,828              66
for the proposed technique. This matrix pertains to the 22                                           1,799
categories of leaf disease images inside the public dataset. The              9 Cherry_Powdery        251               74
majority of the categories have demonstrated encouraging                            Mildew serious    857
outcomes. The leaf disease images that exhibit inadequate                                             770               59
classiﬁcation pertain to the plant species “Apple” and “Citrus.”             10 Grape healthy        11,603
The category labeled as “Citrus healthy” can sometimes be mistaken                                                       9
with the category known as “Citrus Greening June general.” The               11 Grape Black Rot
attribution of the resemblance between various forms of leaf                        Fungus general                      90
diseases is warranted. Another challenging classiﬁcation
assignment involves distinguishing between “Apple_Scab general”              12 Grape Black Rot                         52
and “Apple_Scab serious.” This phenomenon may be ascribed to                        Fungus serious                      269
the existence of two distinct variants of an image falling under the
overarching classiﬁcation of “Apple_Scab.”                                   13 Grape Black Measles                     262
                                                                                    Fungus general
    In addition, the T-distributed stochastic neighbor embedding (t-                                                    36
SNE) was implemented using the suggested methodology, as seen in             14 Grape Black Measles                     122
Figure 8, van der Maaten and Hinton (2008). It should be noted that                 Fungus serious
t-SNE is a computational approach employed for the purpose of                                                           110
visualizing the multidimensional feature space of the 22 categories of       15 Grape Leaf Blight
sick leaves in a two-dimensional (2D) format. Figure 8 presents a                   Fungus general                     1,668
summary of the t-SNE clustering outcomes for both the output
produced by the suggested technique and the ground truth. Figure 8           16 Grape Leaf Blight
exhibits a notable clustering pattern as classes 16 and 17 are closely              Fungus serious
packed together on the right side. It should be noted that the distinct
attributes of these leaf images can only be ascribed to a limited            17 Citrus healthy
number of locations that are outside the clusters.
                                                                             18 Citrus Greening
3.4 Discussion                                                                      June general

    The utilization of CNN models in deep learning has become                19 Citrus Greening
prevalent. These models possess the capability to extract feature                   June serious
maps from images. Furthermore, the effectiveness of feature
extraction may be enhanced by employing a network structure                  20 Peach healthy

                                                                             21 Peach_Bacterial
                                                                                    Spot general

                                                                             22 Peach_Bacterial
                                                                                    Spot serious

                                                                              – Total

                                                                             with increased depth. Nevertheless, the efﬁcacy of CNNs may be
                                                                             limited due to the inherent constraint of the convolutional module,
                                                                             which primarily emphasizes the analysis of small receptive ﬁelds
                                                                             inside the images. This phenomenon rapidly results in the disregard
                                                                             of the interconnections among distant pixels within an image. In
                                                                             addition, the process of enhancing the performance of deeper
                                                                             convolutional neural network models necessitates a greater
                                                                             allocation of processing resources.

Frontiers in Plant Science                                               08                                         frontiersin.org
Han and Guo                                                                         10.3389/fpls.2023.1328952

TABLE 3 Comparison results between the state-of-the-arts and the proposed method.

     Method                               Accuracy                       Precision  Recall   F1 score

  AlexNet                                      78.51                         77.63    80.16      78.87
  GoogleNet                                    81.23                         80.59    82.05      81.31
  VGG                                          82.35                         82.19    82.94      82.56
  ResNet101                                    83.18                         82.56    83.29      82.92
  EfﬁcientNetB3                                83.25                         83.03    83.48      83.25
  Inception V3                                 84.01                         83.23    84.33      83.78
  MobileNet V2140                              84.69                         83.52    84.92      84.21
  Vision Transformer                           85.47                         84.38    86.71      85.53
  Our method                                  86.43                          85.73   87.22       86.47
Bold values denote the best performance.

    In the context of leaf disease images, it is observed that the       suggested dual channel model employs the technique of MSA to
affected regions are frequently dispersed over the whole image,          continually extract the correlation between image patches. This
rather than being conﬁned to a speciﬁc localized location. This          approach effectively preserves the information that is
characteristic is exempliﬁed in Figure 9. Given the limitations of       advantageous for classiﬁcation purposes. In contrast to the
the local receptive ﬁeld mechanism in addressing the speciﬁc leaf        original vision transformer model, the swin vision transformer
disease image, the mere addition of extra layers to the CNN              model is capable of extracting valuable information from images
models does not always ensure improved performance in image              while concurrently mitigating its computing resource
classiﬁcation. This study presents the introduction of a vision          requirements. Nevertheless, this research endeavor is subject to
transformer-based model for image classiﬁcation, which leverages         many constraints: The dataset utilized in the experiments suffers
the relationships among distant pixels inside the images. The            from unbalanced image samples, hence limiting the effectiveness

FIGURE 7

The confusion matrix of the proposed approach on the presented dataset.

Frontiers in Plant Science                                          09                       frontiersin.org
Han and Guo                                                               10.3389/fpls.2023.1328952

FIGURE 8

The outcome of performing t-SNE on outcome generated from the proposed approach (Top) and on the ground truth samples (Bottom).

of the presented method. Meanwhile, number of the image                   intended to capture the edges in the merged image. In addition, the
samples contained in the leveraged dataset is still limited, which        Sobel operator has been utilized to extract the edge information
constrains the accuracy of the proposed approach at relatively low        from the images of leaf diseases. The utilization of the two-channel
level. In addition, there exists duplication between the edge             swin vision transformer model has resulted in the attainment of
information included in the lower channel of the proposed                 improved performance compared to the current state-of-the-art
model and the upper channel.                                              methods. The efﬁcacy of the suggested model is demonstrated by
                                                                          experimental ﬁndings conducted on the publically accessible
4 Conclusion                                                              dataset. The experimental results of the proposed approach have
                                                                          proved the superior performance of the proposed approach in leaf
    The present study introduces a novel network architecture for         disease classiﬁcation. It can be concluded that the proposed
leaf disease image classiﬁcation, utilizing a two-channel swin            approach could be a valuable algorithm for leaf classiﬁcation and
transformer-based approach. The system consists of a dedicated            Precision Agriculture.
channel for the original image and an additional channel speciﬁcally
                                                                              Recently, there has been encouraging performance
                                                                          demonstrated by vision transformer-based models in challenges

Frontiers in Plant Science                                            10                                                         frontiersin.org
Han and Guo                                                                                                         10.3389/fpls.2023.1328952

                                                                                               Data availability statement

                                                                                                   Publicly available datasets were analyzed in this study. The
                                                                                               datasets for this study can be found in the AI challenger 2018 at
                                                                                               https://aistudio.baidu.com/datasetdetail/76075.

                                                                                               Author contributions

                                                                                                   DH: Conceptualization, Data curation, Formal Analysis,
                                                                                               Investigation, Visualization, Writing – original draft. CG:
                                                                                               Funding acquisition, Methodology, Project administration,
                                                                                               Supervision, Validation, Writing – original draft.

                                                                                               Funding

                                                                                                   The author(s) declare that no ﬁnancial support was received for
                                                                                               the research, authorship, and/or publication of this article.

                                                                                               Conﬂict of interest

                                                                                                   The authors declare that the research was conducted in the
                                                                                               absence of any commercial or ﬁnancial relationships that could be
                                                                                               construed as a potential conﬂict of interest.

FIGURE 9                                                                                       Publisher’s note

A sample image contains the diseased areas across the entire image.

related to multi-modal machine vision. Henceforth, we shall further                                All claims expressed in this article are solely those of the authors
explore the intricacies of multi-model-based deep learning models                              and do not necessarily represent those of their afﬁliated
in the context of leaf disease categorization and prediction. In                               organizations, or those of the publisher, the editors and the
addition, more samples need to be collected to eliminate the class                             reviewers. Any product that may be evaluated in this article, or
imbalance issue in the dataset used in this study.                                             claim that may be made by its manufacturer, is not guaranteed or
                                                                                               endorsed by the publisher.

References

  Abas, M. A. H., Ismail, N., Yassin, A. I. M., and Taib, M. N. (2018). Vgg16 for plant          Chen, J., Chen, J., fu Zhang, D., Sun, Y., and Nanehkaran, Y. A. (2020). Using deep
image classiﬁcation with transfer learning and data augmentation. Int. J. Eng. Technol. 7      transfer learning for image-based plant disease identiﬁcation. Comput. Electron. Agric.
(4), 90–94. doi: 10.14419/IJET.V7I4.11.20781                                                   173, 105393. doi: 10.1016/j.compag.2020.105393

  Afzaal, H., Farooque, A. A., Schumann, A. W., Hussain, N., McKenzie-Gopsill, A.,               Devi, K. S. G., Balasubramanian, K., Senthilkumar, C., and Ramya, K. (2022).
Esau, T. J., et al. (2021). Detection of a potato disease (early blight) using artiﬁcial       Accurate prediction and classiﬁcation of corn leaf disease using adaptive moment
intelligence. Remote. Sens. 13, 411. doi: 10.3390/rs13030411                                   estimation optimizer in deep learning networks. J. Electrical Eng. Technol. 18, 637–649.
                                                                                               doi: 10.1007/s42835-022-01205-0
  Anagnostis, A., Asiminari, G., Papageorgiou, E. I., and Bochtis, D. D. (2020). A
convolutional neural networks based method for anthracnose infected walnut tree                  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
leaves identiﬁcation. Appl. Sci. 10 (2), 469. doi: 10.3390/app10020469                         et al. (2020). An image is worth 16x16 words: Transformers for image recognition at
                                                                                               scale. ArXiv. doi: 10.48550/arXiv.2010.11929
  Ba, J., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. ArXiv. doi:
10.48550/arXiv.1607.0645                                                                         Elfatimi, E., Eryigit, R., and Elfatimi, L. (2022). Beans leaf diseases classiﬁcation using
                                                                                               mobilenet models. IEEE Access 10, 1–1. doi: 10.1109/ACCESS.2022.3142817
  Bo, G., Leilei, D., Wei, L., and Bo, L. (2019). Research on maize disease image
recognition method based on grabcut algorithms. J. Chin. Agric. Mechanization. doi:              JencyRubia, J., and BabithaLincy, R. (2021). Detection of plant leaf diseases using
10.1155/2021/9110866                                                                           recent progress in deep learning-based identiﬁcation techniques. J. Eng. Technol. Ind.
                                                                                               Appl. 7, 29–36. doi: 10.5935/jetia.v7i30.768

Frontiers in Plant Science                                                                 11                       frontiersin.org
Han and Guo                                                                                        10.3389/fpls.2023.1328952

  Jenipher, V. N., and Radhika, S. (2022). “An automated system for detecting rice crop              Singh, V. B., and Misra, A. K. (2017). Detection of plant leaf diseases using image
disease using cnn inception v3 transfer learning algorithm,” in 2022 Second                        segmentation and soft computing techniques. Inf. Process. Agric. 4, 41–49. doi: 10.1016/
International Conference on Artiﬁcial Intelligence and Smart Energy (ICAIS).                       j.inpa.2016.10.005
(Piscataway, NJ: IEEE) 88–94.
                                                                                                     Singh, R., Sharma, A., Anand, V., and Gupta, R. (2022). “Impact of efﬁcientnetb3 for
  Kai, S., Zhikun, L., Hang, S., and Chunhong, G. (2011). “A research of maize disease             stratiﬁcation of tomato leaves disease,” in 2022 6th International Conference on
image recognition of corn based on bp networks,” in Third International Conference on              Electronics, Communication and Aerospace Technology. (Piscataway, NJ: IEEE).
Measuring Technology & Mechatronics Automation. (Piscataway, NJ: IEEE).                            1373–1378.

  Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). Imagenet classiﬁcation with               Sneha, H., and Bagal, S. (2019). Detection of plant leaf diseases using image
deep convolutional neural networks. Commun. ACM 6084–90. doi: 10.1145/3065386                      processing. Int. J. Advance Res. Innovative Ideas Educ. 5, 168–170. doi: 10.31838/
                                                                                                   jcr.07.06.310
  Ksibi, A., Ayadi, M., Souﬁene, B., Jamjoom, M. M., and Ullah, Z. (2022). Mobires-net:
A hybrid deep learning model for detecting and classifying olive leaf diseases. Appl. Sci.           Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S. E., Anguelov, D., et al. (2014).
12 (20), 10278. doi: 10.3390/app122010278                                                          “Going deeper with convolutions,” in 2015 IEEE Conference on Computer Vision and
                                                                                                   Pattern Recognition (CVPR). (Piscataway, NJ: IEEE). 1–9.
  Kulkarni, R. R., and Sapariya, A. D. (2021). Detection of plant leaf diseases using machine
learning. 10th International Conference on Computing, Communication and Networking                   Tolstikhin, I. O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T.,
Technologies, (Piscataway, NJ: IEEE). doi: 10.1109/ICCCNT45670.2019.8944556                        et al. (2021). “Mlp-mixer: An all-mlp architecture for vision,” in Neural Information
                                                                                                   Processing Systems. (Cambridge, USA: MIT Press).
  Liu, B., Zhang, Y., He, D., and Li, Y. (2017). Identiﬁcation of apple leaf diseases based
on deep convolutional neural networks. Symmetry 10, 11. doi: 10.3390/sym10010011                     van der Maaten, L., and Hinton, G. E. (2008). Visualizing data using t-sne. J. Mach.
                                                                                                   Learn. Res. 9, 2579–2605.
  Liu, W., and Wang, L. (2022). Quantum image edge detection based on eight-
direction sobel operator for neqr. Quantum Inf. Process. 21, 1–27. doi: 10.1007/s11128-              Wang, G., Wang, J., Yu, H., and Sui, Y. (2022). Research on identiﬁcation of corn
022-03527-4                                                                                        disease occurrence degree based on improved resnext network. Int. J. Pattern
                                                                                                   recognition Artif. Intell. 36. doi: 10.1142/S0218001422500057
  Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., et al. (2021). “Swin transformer:
Hierarchical vision transformer using shifted windows,” in 2021 IEEE/CVF                             Wang, G., Yu, H., and Sui, Y. (2021). Research on maize disease recognition method
International Conference on Computer Vision (ICCV). (Piscataway, NJ: IEEE). 9992–                  based on improved resnet50. Mobile Inf. Syst. doi: 10.1155/2021/9110866
10002.
                                                                                                     Wu, J., Zheng, H., Zhao, B., Li, Y., Yan, B., Liang, R., et al. (2017). Ai challenger: A
  Mahum, R., Munir, H., un-nisa Mughal, Z., Awais, M., Khan, F. S., Saqlain, M., et al.            large-scale dataset for going deeper in image understanding. ArXiv. doi: 10.48550/
(2022). A novel framework for potato leaf disease detection using an efﬁcient deep                 arXiv.1711.06475
learning model. Hum. Ecol. Risk Assessment: Int. J. 29, 303–326. doi: 10.1080/
10807039.2022.2064814                                                                                Xu, Y., Hao, Q., Qiao, L., and Wang, S. (2020). “Study on classiﬁcation of maize
                                                                                                   disease image based on fast k-nearest neighbor support,” in 2020 IEEE Intl Conf on
  Mounika, R., and Bharathi, P. S. (2020). Detection of plant leaf diseases using image            Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and
processing. Agricultural Food Sci.                                                                 Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and
                                                                                                   Technology Congress (DASC/PiCom/CBDCom/CyberSciTech). (Piscataway, NJ: IEEE).
  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019).
Pytorch: An imperative style, high-performance deep learning library. ArXiv. doi:                    Y, V., Billakanti, N., Veeravalli, K. ,. N., A. D., R., and Kota, L. (2022). “Early
10.48550/arXiv.1912.01703                                                                          detection of casava plant leaf diseases using efﬁcientnet-b0,” in 2022 IEEE Delhi Section
                                                                                                   Conference (DELCON). (Piscataway, NJ: IEEE).1–5.
  Patil, P., Yaligar, N., and M, M. (2017). “Comparision of performance of classiﬁers -
svm, rf and ann in potato blight disease detection using leaf images,” in 2017 IEEE                  Yao, J., Wang, Y., Xiang, Y., Yang, J., Zhu, Y., Li, X., et al. (2022). Two-stage detection
International Conference on Computational Intelligence and Computing Research                      algorithm for kiwifruit leaf diseases based on deep learning. Plants 11. doi: 10.3390/
(ICCIC). (Piscataway, NJ: IEEE). 1–5.                                                              plants11060768

  Qian, X., Zhang, C., Chen, L., and Li, K. (2022). Deep learning-based identiﬁcation of             Yu, M., Ma, X., Guan, H., Liu, M., and Tao, Z. (2022). A recognition method of
maize leaf diseases is improved by an attention mechanism: Self-attention. Front. Plant            soybean leaf diseases based on an improved deep learning model. Front. Plant Sci. 13.
Sci. 13. doi: 10.3389/fpls.2022.864486                                                             doi: 10.3389/fpls.2022.878834

  Reddy, P. C., Chandra, R. M. S., Vadiraj, P., Reddy, M. A., Mahesh, T. R., and                     Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., and Yoo, Y. J. (2019). “Cutmix:
Madhuri, G. S. (2021). “Detection of plant leaf-based diseases using machine learning              Regularization strategy to train strong classiﬁers with localizable features,” in 2019
approach,” in 2021 IEEE International Conference on Computation System and                         IEEE/CVF International Conference on Computer Vision (ICCV). (Piscataway, NJ:
Information Technology for Sustainable Solutions (CSITSS). (Piscataway, NJ: IEEE).                 IEEE). 6022–6031.
1–4.
                                                                                                     Zhang, Q. (2021). A novel resnet101 model based on dense dilated convolution for
  Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., et al. (2014).              image classiﬁcation. SN Appl. Sci. 4. doi: 10.1007/s42452-021-04897-7
Imagenet large scale visual recognition challenge. Int. J. Comput. Vision 115, 211–252.
doi: 10.1007/s11263-015-0816-y                                                                       Zhang, H., Cissé, M., Dauphin, Y., and Lopez-Paz, D. (2017). mixup: Beyond
                                                                                                   empirical risk minimization. ArXiv. doi: 10.48550/arXiv.1710.09412

Frontiers in Plant Science                                                                     12  frontiersin.org

---

## EXTRACTED FIGURES (visual analysis)

### Figure 2 - two-channel Swin Transformer (page 4, `papers/16_fpls/inputs/figures/page4.png`)
- Two parallel channels, no weight sharing. Top: input H x W x 3 original RGB image. Bottom: input H x W x 3 edge/fused image (blue-ish).
- Each channel: Patch Partition (H/4 x W/4 x 48) -> Stage 1: Linear Embedding + Swin Transformer Block x2 (H/4 x W/4 x C) -> Stage 2: Patch Merging + Swin Block x2 (H/8 x W/8 x 2C) -> Stage 3: Patch Merging + Swin Block x6 (H/16 x W/16 x 4C) -> Stage 4: Patch Merging + Swin Block x2 (H/32 x W/32 x 8C).
- Both Stage 4 outputs -> Concat -> Average Pooling -> Fully Connected -> class labels (Apple Healthy ... Diseased ones). C=96 (caption). Softmax/cross-entropy per text.
- Not shown in figure: number of heads, window size, MLP ratio, dropout (heads are not stated anywhere in paper; Swin-T standard would be 3/6/12/24 but NOT stated).

### Figure 3 - bottom-channel input formation (page 5, `.../figures/page5.png`)
- Original image -> three parallel branches: (a) Gray Scale Image, (b) 3x3 Conv X-direction (Sobel Gx), (c) 3x3 Conv Y-direction (Sobel Gy). Outputs of the three are "Merging" (shown as stacked as 3 channels of a fused image; the merge is drawn as arrows into one colored image, i.e. channel stacking: gray, Gx, Gy) -> 3-channel input for bottom channel. Text: Sobel conv weights initialised from Gx/Gy and trainable by backprop (ablation compares with fixed Sobel).
- Sobel kernels: Gx=[[+1,0,-1],[+2,0,-2],[+1,0,-1]], Gy=[[+1,+2,+1],[0,0,0],[-1,-2,-1]] (from text eq. 1-2; pdftotext sign layout garbled, standard Sobel).

### Table 1 - Hyper-parameters (page 8, `.../figures/page8.png`)
Batch_size 8 | optimizer Adam | learning rate 1e-4 | depth 12 | epochs 100. (pdftotext garbled the order; verified visually.) "depth 12" is not explained (Swin-T total blocks = 2+2+6+2 = 12 per channel).

### Table 2 - Dataset distribution (page 8)
Class: train/test. 1 Apple healthy 1185/169; 2 Apple_Scab general 211/30; 3 Apple_Scab serious 152/22; 4 Apple Frogeye Spot 427/61; 5 Cedar Apple Rust general 142/20; 6 Cedar Apple Rust serious 40/6; 7 Cherry healthy 598/85; 8 Cherry_Powdery Mildew general 116/12; 9 Cherry_Powdery Mildew serious 110/18; 10 Grape healthy 294/42; 11 Grape Black Rot general 381/54; 12 Grape Black Rot serious 462/66; 13 Grape Black Measles general 503/74; 14 Grape Black Measles serious 419/59; 15 Grape Leaf Blight general 61/9; 16 Grape Leaf Blight serious 630/90; 17 Citrus healthy 367/52; 18 Citrus Greening June general 1828/269; 19 Citrus Greening June serious 1799/262; 20 Peach healthy 251/36; 21 Peach_Bacterial Spot general 857/122; 22 Peach_Bacterial Spot serious 770/110. Total 11603/1668.

### Figure 4 (Swin block), Figs 5-9: not rendered (generic Swin block / SW-MSA diagram / ablation bars / confusion matrix / t-SNE / sample image; no extra reproduction values beyond text).
