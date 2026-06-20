# PAPER CONTENT: 10 - PDLCViT_Experiment

**Source PDF:** E:\UIT\study\claude\learn-claude\papers\10 - PDLCViT_Experiment\inputs\s44196-024-00597-3.pdf
**Extracted:** 2026-06-19

---

International Journal of Computational Intelligence Systems  (2024) 17:188
https://doi.org/10.1007/s44196-024-00597-3

RESEARCH ARTICLE

A Multitask LearningBased Vision Transformer for Plant Disease
Localization and Classification

S. Hemalatha1 · Jai Jaganath Babu Jayachandran2

Received: 25 April 2024 / Accepted: 5 July 2024
© The Author(s) 2024

Abstract

Plant disease detection is a critical task in agriculture, essential for ensuring crop health and productivity. Traditional methods
in this context are often labor-intensive and prone to errors, highlighting the need for automated solutions. While computer
vision-based solutions have been successfully deployed in recent years for plant disease identification and localization
tasks, these often operate independently, leading to suboptimal performance. It is essential to develop an integrated solu-
tion combining these two tasks for improved efficiency and accuracy. This research proposes the innovative Plant Disease
Localization and Classification model based on Vision Transformer (PDLC-ViT), which integrates co-scale, co-attention, and
cross-attention mechanisms and a ViT, within a Multi-Task Learning (MTL) framework. The model was trained and evalu-
ated on the Plant Village dataset. Key hyperparameters, including learning rate, batch size, dropout ratio, and regularization
factor, were optimized through a thorough grid search. Early stopping based on validation loss was employed to prevent
overfitting. The PDLC-ViT model demonstrated significant improvements in plant disease localization and classification
tasks. The integration of co-scale, co-attention, and cross-attention mechanisms allowed the model to capture multi-scale
dependencies and enhance feature learning, leading to superior performance compared to existing models. The PDLC-ViT
model evaluated on two public datasets achieved an accuracy of 99.97%, a Mean Average Precision (MAP) of 99.18%, and
a Mean Average Recall (MAR) of 99.11%. These results underscore the model's exceptional precision and recall, highlight-
ing its robustness and reliability in detecting and classifying plant diseases. The PDLC-ViT model sets a new benchmark in
plant disease detection, offering a reliable and advanced tool for agricultural applications. Its ability to integrate localiza-
tion and classification tasks within an MTL framework promotes timely and accurate disease management, contributing to
sustainable agriculture and food security.

Keywords Plant disease · Classification · Localization · Vision transformer · Multi-task learning

Abbreviations
PLDC-ViT  Plant Disease Localization and Classification Vision Transformer
MAP       Mean Average Precision
MAR       Mean Average Recall
AUC       Area Under the Curve
CNN       Convolutional Neural Network
ViT       Vision Transformer
MTL       Multi-Task Learning
XAI       Explainable Artificial Intelligence
Grad-CAM  Gradient-weighted Class Activation Mapping
GAM       Gradient Activation Map
IoU       Intersection over Union
ACC       Accuracy
Sn        Sensitivity (Recall)
Sp        Specificity
Pre       Precision
F1        F1 Score
MSCVT     Multiscale Convolution and Vision Transformer
CMTL      Concurrent Multi-Task Learning

1 Department of Artificial Intelligence and Machine Learning,
  Rajalakshmi Engineering College, Chennai 602 105, India

2 Department of Biomedical Engineering, Chennai Institute
  of Technology, Chennai 600 069, India

--- PAGE BREAK ---

188  Page 2 of 21  International Journal of Computational Intelligence Systems (2024) 17:188

1 Introduction

Plant diseases can have a severe impact on crop health, leading to substantial damage, reduced yields, and finan-
cial losses for farmers. Addressing these diseases is of utmost importance to ensure food security and sustain-
able agriculture. These diseases encompass various infections, pathogens, and abnormalities that adversely affect
the health and growth of plants. They can be caused by bacteria, fungi, viruses, nematodes, or environmental
factors such as nutrient deficiencies or adverse weather conditions [1]. The timely and precise detection of plant
diseases plays a vital role in implementing effective disease management and prevention strategies.

  Conventional plant disease detection methods heavily depend on manual visual inspection conducted by experts.
However, these methods have several drawbacks, including being time-consuming, subjective, and susceptible to
human errors. In recent years, automated plant disease detection systems have shown great promise in overcom-
ing the limitations of traditional methods [2]. In recent times, there have been significant breakthroughs in deep
learning and computer vision, presenting promising opportunities for automating the detection and classification of
plant diseases [3]. These cutting-edge approaches have transformed the field of plant disease detection, harness-
ing the capabilities of artificial neural networks to extract meaningful features from plant images automatically. This
advanced technology enables accurate and efficient disease classification and detection, revolutionizing the way plant
diseases are diagnosed using images.

  Convolutional Neural Networks (CNNs) [4] have been widely employed for plant disease detection due to their
ability to capture spatial dependencies in images. These networks are built with multiple layers of convolutional
and pooling operations, allowing them to learn hierarchical representations of input images. CNNs have shown
remarkable performance in various plant disease detection tasks, achieving high accuracy and robustness.

  Transfer learning is a commonly used technique in deep learning-based plant disease detection [5]. Pretrained CNN
models, such as VGG [6], ResNet [7], or Inception [8], trained on large-scale image datasets are utilized as fea-
ture extractors. Subsequently, the pretrained models are fine-tuned on plant disease datasets, allowing the learned
features to be tailored and optimized for disease detection tasks. Transfer learning accelerates the training pro-
cess, especially when the available plant disease dataset is limited.

  CNNs, although widely used in computer vision, have limitations in capturing long-range dependencies and rely
heavily on large labeled datasets, making it challenging to capture global context information needed for specific
plant disease detection scenarios. Limited or imbalanced datasets can result in overfitting or biased models,
impacting real-world performance. Additionally, CNNs can be computationally expensive, especially with deeper
architectures, necessitating significant computational resources and training time. Similarly, ResNet architec-
tures encounter limitations such as the vanishing gradient problem with increasing layers, increased memory require-
ments for deeper models, and the potential for overfitting due to added parameters from skip connections. Incep-
tion models, known for their complexity, can suffer from information loss through spatial downsampling, affecting
the retention of fine-grained details crucial for accurate disease detection. They also demand more computational
resources and longer training time while limiting interpret-ability and transparency. The limitations of conventional
deep learning models used for plant disease detection are extensively discussed by Arsenovic et al. [9].

  While deep learning models have achieved remarkable results, they often focus on either disease localization or dis-
ease classification as separate tasks. Integrating both localization and classification tasks into a multitasking framework
can provide a more comprehensive and practical solution for plant disease detection. Empirical studies have demonstrated
that MultiTask Learning (MTL), which involves the simultaneous learning of features from multiple related tasks, offers
significant advantages in prediction performance compared to discrete tasks [10]. MTL mitigates the risk of overfitting
in individual tasks by training them in parallel, enabling the model to utilize a broader range of features from different
tasks. This results in improved generalization capabilities, enhancing the overall performance of the model.

  ViTs are a recent breakthrough in computer vision, adapting the Transformer architecture, originally designed for
natural language processing, to handle image-based tasks [11]. Unlike CNNs, which rely on convolutional operations
for feature extraction, ViTs adopt a self-attention mechanism to capture global dependencies within an image.
This allows ViTs to model relationships between all image patches, enabling them to learn meaningful representations
and achieve state-of-the-art performance in various computer vision tasks. Significant works on MTL models and
ViTs for plant disease detection have been proposed in [12] and [13], respectively.

  This investigation stems from the need for innovative technology driven solutions for sustainable agriculture, and
the limitations of existing models. These models often fail to effectively handle multiple tasks simultaneously, leading
to suboptimal performance in both disease localization and classification. The integration of these two tasks is crucial
for sustainable agriculture as it enables precise, targeted interventions, reducing the need for broad-spectrum pesti-
cides and minimizing environmental impact. This approach can facilitate timely management of plant diseases, thereby
enhancing crop health and productivity. Inspired by the demand for such integral models, and the demonstrated
abilities of MTL frameworks and ViTs, this research aims to build the PLDC-ViT for plant disease localization and clas-
sification. The contributions of this research are as below:

1. ViT-based MTL framework leveraging the power of ViTs to capture global dependencies and encode local
   and global information from plant images.

2. Joint localization and classification by jointly considering disease localization and classification as intercon-
   nected tasks for a comprehensive and accurate analysis of plant diseases, leading to improved performance and
   more effective disease management.

  The rest of this paper is structured as follows: Sect. 2 provides a review of related works in this research domain. Sec-
tion 3 describes the datasets employed and methods used in this study. In Sect. 4, the proposed problem, the PDLC-ViT
framework, and the training procedure are discussed. Section 5 presents the empirical results, comparative analyses,
and the key findings of the study. Finally, Sect. 6 concludes the paper by summarizing the contributions, highlighting the
research's significance, potential applications, and opportunities for future extensions.

2 Related Works

[...Related works section summarizing prior CNN, MTL, and ViT approaches for plant disease detection - see full text for details...]

Key related works mentioned:
- Thakur et al. [13]: PlantViT (CNN+ViT hybrid with multi-head attention), 98.61% on PlantVillage, 87.87% on Embrapa
- Thai et al. [24]: ViT for cassava leaf disease, Acc 98.86%
- Thakur et al. [25]: PlantXViT (lightweight CNN+ViT), >93.55%/92.59%/98.33% on Apple/Maize/Rice
- Zhu et al. [26]: MSCVT (multiscale conv + ViT), Acc 99.86%
- Li et al. [27]: ConvViT (conv + Transformer), Acc 96.85%
- Li et al. [28]: SLViT (shuffle-conv + Transformer encoder), Plant Village SOTA
- Goncalves et al. [33]: MTLSegFormer (MTL+Transformer semantic segmentation), Leaf F1: 0.9869, IoU: 0.9743

--- PAGE BREAK ---

3 Materials and Methods

3.1 Dataset

The PDLC-ViT is trained and tested with the PlantVillage [34] and the PlantDoc [35] datasets. The PlantVillage data-
set contains 54,306 healthy and unhealthy leaf images categorized into 38 different classes based on species and dis-
ease. It was curated by researchers at Penn State University to support the advancement of mobile disease diagnostics
through machine vision approaches. The public dataset provides researchers and developers with a valuable resource
for exploring and evaluating machine learning techniques in automated plant disease detection and classification in
agriculture. The distribution of the plant classes for training and testing is given in Table 1. The original dataset is
augmented with images by affine transformation from the original images. The Plantdoc dataset consists of an aug-
mented dataset of 2940 images collected from 13 species with corresponding segmentation masks. In this research,
this dataset is used to evaluate the disease localization ability of the model.

Table 1 - Distribution of classes in plant village dataset

Leaf               Infection          Training    Testing
Apple              Scab               630         630
                   Black_rot          621         621
Blackberry         Rust               275         275
Cherry             Healthy            1645        1645
Corn_(maize)       Healthy            1502        1502
                   Healthy            854         854
Grape              Powdery_mildew     1052        1052
                   Gray_leaf_spot     513         513
Citrus             Common_rust        1192        1192
Peach              Healthy            1162        1162
Pepper_Bell        Leaf Blight        985         985
Potato             Black_rot          1180        1180
Raspberry          Black_Measles      1384        1384
Soybean            Healthy            423         423
Squash             Leaf Blight        1076        1076
Strawberry         Greening           5507        5507
Tomato             Bacterial_Spot     2297        2297
                   Healthy            360         360
Total              Bacterial_Spot     997         997
                   Healthy            1478        1478
                   Early Blight       1000        1000
                   Healthy            152         152
                   Late Blight        1000        1000
                   Healthy            371         371
                   Healthy            5090        5090
                   Powdery_mildew     1835        1835
                   Healthy            456         456
                   Leaf_scorch        1109        1109
                   Bacterial_Spot     2127        2127
                   Early_Blight       1000        1000
                   Healthy            1591        1591
                   Late Blight        1909        1909
                   Leaf Mold          952         952
                   Leaf Spot          1771        1771
                   Spider Mite        1676        1676
                   Target Spot        1404        1404
                   Mosaic Virus       373         373
                   Leaf Curl Virus    5357        5357
                              Total:  54,306      54,306

3.2 Details of Implementation

The PDLC-ViT is implemented using Matlab 2023a with deep learning and image processing toolboxes. The training
and testing occur on a high-performance system with an i7 processor powered by 16 GB DDR4 RAM, and NVIDIA
GeForce RTX 3070 8 GB Graphics card for efficient computation. This setup ensures accurate plant disease detection
and classification performance assessment.

--- PAGE BREAK ---

4 Proposed PDLC-ViT Model

This section gives a mathematical description of the joint segmentation and classification problem addressed in this
research, detailed architecture of the proposed model and its components and the training procedure.

4.1 Problem Formulation

Given a dataset D = (xi, yi)^N_{i=1}, where xi represents the ith input image and yi represents the corresponding ground
truth label, the problem of disease localization and classification in plant leaf images can be formulated as follows.
Let X ∈ R^{H×W×C} represent the input image tensor, where H, W, and C denote the height, width, and number of channels,
respectively. The disease localization task aims to predict a binary mask M ∈ {0,1}^{H×W} that indicates the presence
or absence of the disease in each pixel of the image. The disease classification task involves predicting the class
label L ∈ {1,2,...,K}, where K is the total number of disease classes.

  To jointly address the disease localization and classification tasks, an MTL model is proposed. This model con-
sists of a shared feature extractor f_shared and task-specific modules, f_loc for localization and f_cls for classification.
The shared feature extractor maps the input image X to a shared feature representation Z ∈ R^{H×W×d}, where H, W,
and d represent the dimensions of the feature map. The localization module takes the shared feature representa-
tion Z and produces the disease mask M using a mapping function f_loc: R^{H×W×d} → {0,1}^{H×W}. Similarly, the clas-
sification module utilizes the shared features Z and predicts the disease class label L using a mapping function
f_cls: R^{H×W×d} → {1,2,...,K}.

  The MTL model is trained by minimizing a combination of loss functions for both tasks. The disease localization
loss L_loc measures the pixel-wise discrepancy between the predicted mask M and the ground truth mask. This can be
expressed as in (1), where L_loc is a loss function and M_gt denotes the ground truth mask:

    L_loc = (1/HW) * sum_{i=1}^{H} sum_{j=1}^{W} L_loc(M_ij, M_gt_ij)     (1)

  The disease classification loss L_cls measures the discrepancy between the predicted class label L and the ground
truth label y. This can be expressed as in (2) where L_cls is the categorical cross-entropy loss:

    L_cls = L_cls(L, y)     (2)

  The overall loss function for the MTL model is defined as in (3), where λ_loc and λ_cls are hyperparameters that control
the importance of each task during training. The MTL model is trained by minimizing the loss using the Adam optimiza-
tion algorithm. By jointly optimizing the disease localization and classification tasks, the proposed MTL model aims to
improve the accuracy and efficiency of disease detection in plant leaf images, leveraging the shared information between
the tasks for enhanced performance:

    L_MTL = λ_loc * L_loc + λ_cls * L_cls     (3)

4.2 PDLC-ViT High-Level Architecture

The PDLC-ViT is implemented with two parallel branches, one each for segmentation and classification, each compris-
ing the co-scale and co-attention layers, respectively. The co-scale layer is strategically selected to capture multi-scale
dependencies, which are pivotal for accurately detecting disease patterns that can vary significantly in size. This layer
ensures robust feature extraction, enhancing the model's ability to generalize across different plant diseases. The co-
attention layer further refines the representation of image patches by emphasizing spatial relationships, effectively
integrating both local details and global context. This dual focus is crucial for precise disease localization and classi-
fication. Further, the cross-attention mechanism serves as a sophisticated integrator, facilitating seamless informa-
tion exchange between the segmentation and classification tasks. By combining these enriched representations,
the model achieves a comprehensive understanding of the input image, significantly improving overall performance.
This integrated, multi-faceted approach ensures a holistic analysis, leading to more accurate, efficient, and reliable
plant disease detection.

  Given an input image I to a patch splitter, it is divided into a set of non-overlapping patches X = {x1, x2, ..., xn}.
The co-scale layer captures dependencies and interactions at different spatial scales between the patches and produces
the output CS = {xs1, xs2, ..., xsn}, where xsi represents the co-scaled representation of the corresponding patch xi.
The co-scaled representation for each patch xi captures the cross-scale interactions and dependencies between different
patches, enabling a comprehensive understanding of the image content. The co-scale representation CS is utilized as
input to the co-attention mechanism, which generates an output that captures the interdependencies between the patches.

This process enhances the representation by attending to the relationships between different patches. The output of the
co-attention mechanism, denoted as CA = {a1, a2, ..., an}, consists of attended representations that incorporate global
context and cross-scale interactions. These attended representations can then be further processed by subsequent mod-
ules in the architecture, such as the encoder and decoder, to extract high-level features and generate outputs such as
segmentation masks or classification predictions.

  The co-scale and co-attention representations obtained from the segmentation and classification branches of PDLC-
ViT are provided as input to the cross-attention module. This module facilitates the integration of these representations,
enabling the exchange of information between the segmentation and classification tasks. By leveraging both the local and
spatial features captured by the co-scale layer and the global context and cross-scale interactions captured by the co-atten-
tion layer, the cross-attention module combines and fuses the representations to create a comprehensive and enriched
representation Z = {z1, z2, ..., zn}. This integrated representation benefits both the segmentation and classification tasks
by incorporating localized disease information, and capturing dependencies between different regions. It serves as a
basis for generating accurate and detailed outputs, such as segmentation masks and classification predictions.

  The shared feature patches are given as input to the Transformer Encoder which has a stack of transformer layers.
These layers leverage self-attention mechanisms to capture local and global contextual information from Z. The Trans-
former encoder output is fed into the segmentation head, which generates a segmentation map representing pixel-level
disease segmentation. The classification head receives the outputs of the Transformer encoder and the segmentation
head as input. It performs disease classification based on the combined information, providing predictions for the
disease classes. The high-level architecture of the PLDC-ViT is illustrated with Fig. 1. In this architecture, the cross-
attention mechanism combines convolutional layers and self-attention layers to leverage the strengths of both con-
volutional and attention mechanisms to capture local and global dependencies effectively, which offers a good trade-
off between efficiency and performance.

  The Transformer encoder is a critical component of the PDLC-ViT model for plant disease localization and clas-
sification. It takes the shared representation from the cross-attention mechanism, and processes it through a series of
layers to produce the encoded features. The Transformer encoder is depicted with Fig. 2 and the encoding process is
described as below.

--- PAGE BREAK ---

188  Page 8 of 21  International Journal of Computational Intelligence Systems (2024) 17:188

[Fig. 1 - PDLC-ViT high-level architecture - see EXTRACTED FIGURES below]

1. Linear Projection: The input is first linearly projected. This step transforms the shared representation into a
   higher dimensional space suitable for the self-attention mechanism.

2. Positional Encoding: After the linear projection, positional encodings are added to the input. Since the self-
   attention mechanism is not inherently sensitive to the sequence order, positional encodings provide necessary
   information about the position of each element in the sequence.

3. Multi-head Self-attention: The core of the Transformer Encoder, the multi-head self-attention layer, allows the
   model to focus on different parts of the input sequence, capturing complex dependencies. Each "head" computes
   an attention score, indicating the importance of other parts of the input when encoding a particular piece.

4. Add & Norm (Addition and Normalization): The output from the self-attention mechanism undergoes a residual
   connection, which adds the input of the layer before the self-attention to its output to help mitigate the vanishing
   gradient problem. It is then normalized to stabilize the learning process.

5. Feed-Forward: This is a fully connected neural network that processes the output of the attention layer separately
   for each position. It consists of two linear transformations with a Rectified Linear Unit (ReLU) activation
   function in between.

6. Add & Norm: Again, the output from the feed-forward layer is added to its input before the layer (residual con-
   nection) and then normalized. This step is similar to the earlier Add & Norm and ensures that the model can
   learn effectively over many layers without running into training difficulties.

  The final output of the Transformer Encoder is a set of encoded features that are passed on to the subsequent
parts of the model, such as the segmentation head and the classification head, for further processing.

--- PAGE BREAK ---

[Page 9 - Fig. 2 Transformer encoder and Fig. 3 Cross-attention mechanism - see EXTRACTED FIGURES below]

4.3 Cross-attention Mechanism

The cross-attention mechanism is modeled to create the shared representation Z from the co-scaled and co-atten-
tion representations CS and CA, respectively. This mechanism is illustrated in Fig. 3 which consists of two parallel
branches. Initially, CS and CA are transformed into two triples comprising a Query (Q), Key (K), and Value (V) rep-
resented as (Qs, Ks, Vs) and (QA, KA, VA), respectively. Each triple is given as input to the Multi-Head Self-attention
(MHSA) block in each branch which computes the attention scores ASS and ASA as in (4) and (5), where ds and dA
refer to the dimensions of Ks and KA, respectively:

    ASS = softmax(Qs * Ks^T / sqrt(ds)) * Vs     (4)
    ASA = softmax(QA * KA^T / sqrt(dA)) * VA     (5)

  In each branch, the attention scores and keys are passed as input to an Add&Norm block which applies an additive
operation and Layer Normalization (LN) to enhance the feature representation. The output of this block is given as
input to the feed-forward block followed by an Add&Norm block. The outputs Z1 and Z2 are generated from the first and
second branches, respectively, which incorporate both local and global context, as well as the dependencies between
different regions. The shared feature Z is constructed by a Multi-Head Cross-Attention (MHCA) mechanism, which
leverages the complementary strengths of both branches to capture comprehensive information and interdependencies
between different regions. This integration and fusion of the information from Z1 and Z2 result in a shared feature rep-
resentation denoted as Z.

4.4 Segmentation and Classification

The PDLC-ViT performs segmentation and classification tasks with the segmentation and classification heads as
shown in Fig. 4. The segmentation head comprises three blocks within residual connections. The first block consti-
tutes an LN and an MHSA mechanism. The second block comprises an LN block and an MHCA, and the third block
comprises the LN and a Multi-Layer Perceptron (MLP). In the segmentation head, the query tokens of the ground truth
masks are given as input to the first block to capture the dependencies within the ground truth masks.

  The second block of the segmentation head accepts the dependencies captured by the first block, as well as Z, to
capture the interdependencies between the shared feature Z and the ground truth. This enables the model to effec-
tively incorporate both the learned representations from the shared feature and the ground truth information. In the
third block, the MLP layer, which follows the LN, consists of multiple Fully Connected (FC) layers with the Recti-
fied Linear Unit (ReLU) activation function. This function allows the segmentation head to capture high-level
features and relationships within the data, enabling it to extract complex patterns and make accurate predictions
for segmentation tasks.

  Similarly, the classification head comprises two blocks. The first block constitutes the LN and MHCA. The query
tokens of the ground truth class labels corresponding to the segmented image, predicted mask, and the shared
feature Z are given as input to this block. The MHCA mechanism allows the classification head to capture the
interdependencies between the ground truth class labels and the shared feature representation Z. By attending to
these dependencies, the model can effectively incorporate both the learned features from Z and the class label infor-
mation to make accurate predictions. The second block comprises the MLP layer, which follows the LN, where
the MLP consists of a sequence of FC layers with the non-linear sigmoid activation function. It captures high-level
features and relationships within the data, allowing the classification head to extract complex patterns and make
accurate predictions.

4.5 PDLC-ViT Training

The training process of the PDLC-ViT model involves optimizing the parameters to minimize the loss between
the predicted outputs and the ground truth labels. The training procedure is given as below.

Table 2 - PDLC-ViT hyperparameters

Parameter                            Value
Initial Learning Rate                0.001
Learning Rate Reduction              Factor of 0.5 every 15 epochs
Validation Loss Threshold            0.001 (indicating convergence)
Optimizer                            Adam
Mini-Batch Size                      16
Dropout Ratio                        0.2 (applied to both encoder and decoder)
L2 Regularization Factor (λ)         0.0005
Data Handling Strategy               Random shuffling during training and testing
Parameter Optimization               Grid search for optimal parameter configuration

[Full training algorithm from Fig. 11/Page 11 - see EXTRACTED FIGURES section below]

--- PAGE BREAK ---

5 Experimental Results and Discussion

5.1 Experimental Setup

The proposed PDLC-ViT is evaluated using the Plant Village dataset as described in Sect. 3.1. The initial learning
rate of the model is set to 0.001 which undergoes a gradual reduction by a factor of 0.5 every 15 epochs. Model training
is terminated when the validation loss reaches a threshold of 0.001, indicating convergence. The Adam optimizer is
employed to optimize the model, utilizing a mini-batch size of 16. Both the encoder and decoder incorporate a drop-
out ratio of 0.2, while an L2 regularization factor (λ) of 0.0005 is applied across all layers. To prevent overfitting,
the training and testing processes employ a random shuffling strategy. Optimal model parameters are determined
through a thorough grid search, exploring various parameter configurations.

  The number of epochs is not fixed to ensure optimal convergence and prevent overfitting. Instead, early stop-
ping based on the validation loss threshold of 0.001 is employed. This approach allows training to terminate
when the model has sufficiently converged, avoiding unnecessary training and overfitting, and ensuring that
the model achieves the best performance on the validation set. A thorough grid search is conducted to determine
the optimal hyperparameters. This involves systematically varying parameters, such as the learning rate, batch size,
dropout ratio, and regularization factor, and evaluating the model's performance on the validation set. It ensures that
the chosen hyperparameters yield the best possible performance, contributing to the robustness and reliability of
the PDLC-ViT model. Table 2 lists the hyperparameters of the PDLC-ViT model.

5.2 Performance Evaluation

The disease classification and segmentation performance of the PLDC-ViT model is evaluated on the testing dataset
with the following objective metrics. Accuracy (ACC), Sensitivity (Sn), Specificity (Sp), Precision (Pre) and F1
metrics defined with Eqs. (6–9) are evaluated from the True-Positive (TP), True-Negative (TN), False-Positive
(FP), and False-Negative (FN) values resulting from evaluating the model on the test dataset for classification.

    Acc = (TP + TN) / (TP + FP + TN + FN)     (6)
    Sn  = TP / (FN + TP)                       (7)
    Sp  = TN / (FP + TN)                       (8)
    Pre = TP / (TP + FP)                       (9)
    F1  = 2 * Pre * Rec / (Pre + Rec)          (10)

  Segmentation metrics:
    Dice = 2|P ∩ G| / (|P| + |G|)             (11)
    IoU  = |P ∩ G| / |P ∪ G|                  (12)

  MAP (Mean Average Precision):
    MAP = (1/Q) * sum_{q=1}^{Q} AP_q          (13)
    AP  = sum_n (R_n - R_{n-1}) * P_n         (14)

  MAR (Mean Average Recall):
    MAR = (1/Q) * sum_{q=1}^{Q} AR_q          (15)
    AR  = sum_n (P_n - P_{n-1}) * R_n         (16)

5.3 Disease Classification

The classification performance of the model is evaluated under 10 cases as in Table 3. The performance metrics are
given in Table 4 for each classification type.

Table 3 - Plant disease classification types (from visual reading - see EXTRACTED FIGURES)

Classification Type | Crop      | No. of classes | Target classes
A                   | All       | 2              | Healthy, Unhealthy
B                   | Apple     | 4              | Scab, BlackRot, Rust, Healthy
                    |           | 2              | Mildew, Healthy
C                   | Cherry    | 4 / 4          | LeafSpot, Rust, LeafBlight, Healthy / BlackRot, Measles, LeafBlight, Healthy
D                   | Corn      | 2              | BacterialSpot, Healthy
E                   | Grape     | 2 / 3          | BacterialSpot, Healthy / EarlyBlight, LateBlight, Healthy
F                   | Peach     | 2              | LeafScorch, Healthy
G                   | Pepper    | 10             | Bacterial_Spot, Early_Blight, Healthy, Late Blight, Leaf Curl Virus, Leaf Mold, LeafSpot, SpiderMite, TargetSpot, Mosaic Virus
H                   | Potato    |                |
I                   | Strawberry|                |
J                   | Tomato    |                |

Table 4 - Plant disease classification performance metrics (from visual reading - see EXTRACTED FIGURES)

Classification Type | Accuracy | Sensitivity | Specificity | Precision | F1 Score | AUC
A                   | 0.9997   | 0.9996      | 0.9998      | 0.9995    | 0.9995   | 0.9993
B                   | 0.9909   | 0.9907      | 0.9968      | 0.9865    | 0.9885   | 0.9861
                    |          | 0.9941      | 0.9971      | 0.9965    | 0.9953   | 0.9940
C                   | 0.9958   | 0.9921      | 0.9974      | 0.9907    | 0.9914   | 0.9903
                    |          | 0.9863      | 0.9965      | 0.9902    | 0.9882   | 0.9860
D                   | 0.9922   | 0.9861      | 0.9983      | 0.9889    | 0.9875   | 0.9860
E                   | 0.9902   | 0.9970      | 0.9973      | 0.9960    | 0.9965   | 0.9961
F                   | 0.9966   | 0.9871      | 0.9949      | 0.9750    | 0.9809   | 0.9821
G                   | 0.9972   | 1.0         | 0.9982      | 0.9956    | 0.9978   | 0.9971
H                   | 0.9898   | 0.9781      | 0.9984      | 0.9788    | 0.9784   | 0.9781
I                   | 0.9987   |             |             |           |          |
J                   | 0.9858   |             |             |           |          |

Overall: MAP = 99.18%, MAR = 99.11% (evaluated at thresholds 0.3, 0.5, and 0.7)

5.4 Disease Localization

The ability of the model to segment the infected regions for disease localization is evaluated with the Plantdoc
dataset using the segmentation metrics. The model accurately segments:
- Anthracnose: Dice = 0.9945
- Powdery Mildew: Dice = 0.9987
- Early Blight: Dice = 0.9965
- Late Blight: Dice = 0.9973

Mean Dice = 0.9921, Mean IoU = 0.9973 on the test dataset.

5.5 Comparative Analysis

Table 5 - Comparative analysis (from visual reading - see EXTRACTED FIGURES)

Model                     | Dataset                  | Performance metrics                         | Model characteristics
Multi-input Network [21]  | Plant Village            | Acc: 98%, Sp: 99%, Sn: 96%, Pre: 98%       | MTL, shared representations, multi-input network
CMTL [22]                 | Plant Village            | F1: 97%, Acc: 94.52%                        | Jointly learns species and disease classes
ViT [24]                  | Cassava Leaf Disease     | 1%>CNN models, Acc: 98.86%                  | Deployed on Raspberry Pi 4
PlantXViT [25]            | Plant Village            | Acc: 98.86% (Plant Village)                 | Explainable lightweight CNN+ViT
MSCVT [26]                | Plant Village            | Acc: 99.86%                                 | Multiscale conv + self-attention
ConvViT [27]              | Private apple dataset    | Acc: 96.85%                                 | Conv + Transformer structures
SLViT [28]                | Plant Village            | Acc: 98.84%                                 | Transformer plugins in CNN
MTLSegFormer [33]         | Defoliation+Leaf         | Leaf F1: 0.9869, IoU: 0.9743 / Defoliation F1: 0.8877, IoU: 0.8048 | MTL-based ViT for semantic segmentation
PDLC-ViT (Proposed)       | Plant Village + Plantdoc | Acc: 99.97%, Mean Dice: 0.9921, Mean IoU: 0.9973 | Co-scale, co-attention, cross-attention ViT for joint segmentation and classification

5.6 Ablation Study

Table 6 - Performance metrics under ablation study (from visual reading - see EXTRACTED FIGURES)

Model                                    | Binary classification            | Multiclass classification
                                         | Acc   | Sn     | Sp     | Pre    | F1     | Mean Acc | Mean Sn | Mean Sp | Mean Pre | Mean F1
PDLC-ViT (Baseline)                      | 0.9997 | 0.9996 | 0.9998 | 0.9995 | 0.9995 | 0.9937   | 0.9911  | 0.9975  | 0.9898   | 0.9904
PDLC-ViT without co-attention & co-scale | 0.9497 | 0.9496 | 0.9498 | 0.9495 | 0.9495 | 0.9241   | 0.9217  | 0.9276  | 0.9205   | 0.9211
PDLC-ViT without Segmentation Head      | 0.8897 | 0.8896 | 0.8898 | 0.8896 | 0.8896 | 0.8546   | 0.8524  | 0.8578  | 0.8512   | 0.8517

5.7 Explainable Analysis

Grad-CAM (Gradient-weighted Class Activation Mapping) [36] is used to generate Gradient Activation Maps (GAMs)
for analyzing PDLC-ViT and explaining its decisions. The model localizes infections with high Dice coefficients
as shown in Fig. 7 (Dice values: 0.9987, 0.9921, 0.9940, 0.9963).

5.8 Discussion

PDLC-ViT achieves Acc: 99.97% on Plant Village and Mean Dice: 0.9921, Mean IoU: 0.9973 on Plantdoc, outperforming
all compared models. PDLC-ViT surpasses MSCVT (99.86%) by 0.11% in accuracy and MTLSegFormer by 19% in IoU.

Limitation: model is trained with different datasets for segmentation and classification tasks (no single integrated
dataset for end-to-end training), which can lead to domain shifts.

6 Conclusions

The PDLC-ViT model based on joint learning of segmentation and classification tasks for plant disease classification.
Tested on PlantVillage and Plantdoc, demonstrating superior binary and multiclass classification performances.
Mean Dice = 0.9921, Mean IoU = 0.9973, Acc = 99.97%. The model uses MATLAB 2023a.

---

## EXTRACTED FIGURES (visual analysis)

### Figure 1 — PDLC-ViT High-Level Architecture (page 8, `papers/10 - PDLCViT_Experiment/inputs/figures/page8.png`)

Top-down flow diagram:

```
Input Image
    |
Image Splitter
    |
Input Patches: X = {x1, x2, ..., xn}
    |          \
    |           \
[Co-scale      [Co-attention
 Mechanism]     Mechanism]
    |               |
Co-scaled repr.  Co-attention repr.
CS = {xs1, xs2, ..., xsn}   CA = {a1, a2, ..., an}
    |               |
    +---[Cross Attention Mechanism]---+
                    |
             Shared representation
             Z = {z1, z2, ..., zn}
                    |
            [Transformer Encoder]
                    |
           [Segmentation Head] ------> Segmentation Mask
                    |
           [Classification Head] ----> Predicted Labels
```

Color coding in figure:
- Co-scale Mechanism: pink/salmon box
- Co-attention Mechanism: green box
- Cross Attention Mechanism: teal/blue box
- Transformer Encoder: light blue box
- Segmentation Head: pink box
- Classification Head: red/orange box

### Figure 2 — Transformer Encoder (page 9, `papers/10 - PDLCViT_Experiment/inputs/figures/page9.png`)

Vertical stack (bottom to top), processing input shared representation:

```
Shared Representation (input)
        |
[Linear Projection]
        |
[Positional Encoding]
        |
[Multi-Head Self-Attention]
        |
[Add & Norm 1]
        |
[Feed Forward]
        |
[Add & Norm 2]
        |
Encoded Features (output)
```

Each block is shown as a distinct colored rectangle. The flow is strictly sequential (no skip connections shown in the diagram itself, though Add & Norm implies residual connections textually).

### Figure 3 — Cross-attention Mechanism (page 9, `papers/10 - PDLCViT_Experiment/inputs/figures/page9.png`)

Two parallel branches (left branch for CS, right branch for CA), then merged:

Left branch (CS input):
```
CS (co-scaled representation)
    |
[MHSA] <- Ks, Vs (Keys and Values from CS), Qs (Query from CS)
    |
[Add & Norm]
    |
[FeedForward]
    |
[Add & Norm]
    |
Z1
```

Right branch (CA input):
```
CA (co-attention representation)
    |
[MHSA] <- KA, VA (Keys and Values from CA), QA (Query from CA)
    |
[Add & Norm]
    |
[FeedForward]
    |
[Add & Norm]
    |
Z2
```

Merge at top:
```
Z1, Z2 -----> [MHCA] -----> Z (Shared Representation output)
```

MHCA = Multi-Head Cross-Attention, combines Z1 and Z2 to produce shared feature Z.

### Figure 4 — Segmentation and Classification Heads (page 9, `papers/10 - PDLCViT_Experiment/inputs/figures/page9.png`)

Vertical layout with two separate head branches sharing input Z:

Segmentation Head (left branch):
```
Query Tokens of Ground Truth Masks
            |
     Block 1: [LN] -> [MHSA]
            |
     Block 2: [LN] -> [MHCA] <- Z (Shared Features)
            |
     Block 3: [LN] -> [MLP (FC layers + ReLU)]
            |
     Predicted Mask (output)
```

Classification Head (right branch):
```
Query Tokens of Ground Truth Class Labels
(including segmented image, predicted mask, shared feature Z)
            |
     Block 1: [LN] -> [MHCA] <- Z (Shared Features)
            |
     Block 2: [LN] -> [MLP (FC layers + Sigmoid activation)]
            |
     Predicted Labels (output)
```

Both heads have residual connections (shown as arrows bypassing blocks in the figure).

### Table 2 — PDLC-ViT Hyperparameters (page 10, `papers/10 - PDLCViT_Experiment/inputs/figures/page10.png`)

Visually confirmed from page 10 PNG — values match pdftotext exactly:

| Parameter                      | Value                                         |
|-------------------------------|-----------------------------------------------|
| Initial Learning Rate          | 0.001                                         |
| Learning Rate Reduction        | Factor of 0.5 every 15 epochs                 |
| Validation Loss Threshold      | 0.001 (indicating convergence)                |
| Optimizer                      | Adam                                          |
| Mini-Batch Size                | 16                                            |
| Dropout Ratio                  | 0.2 (applied to both encoder and decoder)     |
| L2 Regularization Factor (λ)   | 0.0005                                        |
| Data Handling Strategy         | Random shuffling during training and testing  |
| Parameter Optimization         | Grid search for optimal parameter configuration|

### Training Algorithm (page 11, `papers/10 - PDLCViT_Experiment/inputs/figures/page11.png`)

Input:
- Image dataset: I_dataset = {I_1, I_2, ..., I_n}
- Ground masks: G_masks = {Mask_1, Mask_2, ..., Mask_k}
- Ground truth class labels: G_labels = {Label_1, Label_2, ..., Label_n}
- Number of training iterations: N
- Weighting factors for segmentation and classification losses: λ_cls, λ_seg

Output: Trained PDLC-ViT model

1. Initialize the parameters of the PDLC-ViT model, including the segmentation and classification heads.

2. Repeat for i = 1 to N:
   a. Forward Pass:
      - Obtain the shared feature representation Z from the CoAT
      - Segmentation Head:
        i. First Block:
           - Compute the query tokens of the ground truth masks:
             Q_masks = Embedding(G_masks)
           - Capture the dependencies within the ground truth masks:
             A_masks = MHSA(LN(Q_masks))
        ii. Second Block:
           - Capture the dependencies, predicted mask, and shared feature:
             A_seg = MHCA(LN(A_masks), Z)
        iii. Third Block:
           - Generate the segmentation predictions:
             M_seg = MLP_seg(LN(A_seg))

      - Classification Head:
        i. First Block:
           - Compute the query tokens of the ground truth class labels:
             Q_labels = Embedding(G_labels)
           - Capture the dependencies, predicted mask, and shared feature:
             H_cls = MHCA(LN(Q_labels), Z)
        ii. Second Block:
           - Capture the dependencies, predicted labels, and shared features:
             Hlcls = MHCA(LN(H_labels), Mreg, Z)
        iii. Third Block:
           - Generate the classification predictions:
             P_cls = MLP_cls(LN(H_cls))

   b. Loss Calculation:
      - Compute the segmentation loss:
        L_seg = SegmentationLoss(M_seg, G_masks)
      - Compute the classification loss:
        L_cls = ClassificationLoss(P_cls, G_labels)
      - Compute the total loss:
        L_total = λ_seg * L_seg + λ_cls * L_cls

   c. Backward Pass:
      - Compute gradients of the total loss with respect to the parameters of the segmentation head:
        ∂L_total / ∂θ_seg
      - Compute gradients of the total loss with respect to the parameters of the classification head:
        ∂L_total / ∂θ_cls
      - Update the parameters of the segmentation head using an optimization algorithm:
        θ_seg = Optimize(θ_seg, ∂L_total/∂θ_seg)
      - Update the parameters of the classification head using an optimization algorithm:
        θ_cls = Optimize(θ_cls, ∂L_total/∂θ_cls)

4. End For

### Table 3 — Plant Disease Classification Types (page 12, `papers/10 - PDLCViT_Experiment/inputs/figures/page12.png`)

Visually confirmed full table:

| Classification Type | Crop       | No. of classes | Target classes                                                                                                         |
|--------------------|------------|----------------|------------------------------------------------------------------------------------------------------------------------|
| A                  | All        | 2              | Healthy, Unhealthy                                                                                                     |
| B                  | Apple      | 4              | Scab, BlackRot, Rust, Healthy                                                                                          |
|                    |            | 2              | Mildew, Healthy                                                                                                        |
| C                  | Cherry     | 4              | LeafSpot, Rust, LeafBlight, Healthy                                                                                    |
|                    |            | 4              | BlackRot, Measles, LeafBlight, Healthy                                                                                 |
| D                  | Corn       | 2              | BacterialSpot, Healthy                                                                                                 |
|                    |            | 2              | BacterialSpot, Healthy                                                                                                 |
| E                  | Grape      | 3              | EarlyBlight, LateBlight, Healthy                                                                                       |
|                    |            | 2              | LeafScorch, Healthy                                                                                                    |
| F                  | Peach      | 10             | Bacterial_Spot, Early_Blight, Healthy, Late Blight, Leaf Curl Virus, Leaf Mold, LeafSpot, SpiderMite, TargetSpot, Mosaic Virus |
| G                  | Pepper     | 2              | (2 classes)                                                                                                            |
| H                  | Potato     | 2              | (2 classes)                                                                                                            |
| I                  | Strawberry | 2              | (2 classes)                                                                                                            |
| J                  | Tomato     | 10             | Bacterial_Spot, Early_Blight, Healthy, Late Blight, Leaf Curl Virus, Leaf Mold, LeafSpot, SpiderMite, TargetSpot, Mosaic Virus |

NOTE: The visual rendering of page 12 shows Table 3 clearly. Row mapping to types A-J is:
- A: All crops, 2 classes (Healthy, Unhealthy)
- B: Apple, 4 classes (Scab, BlackRot, Rust, Healthy)
- C: Cherry, 2 classes (Mildew, Healthy)
- D: Corn, 4 classes (LeafSpot, Rust, LeafBlight, Healthy)
- E: Grape, 4 classes (BlackRot, Measles, LeafBlight, Healthy)
- F: Peach, 2 classes (BacterialSpot, Healthy)
- G: Pepper, 2 classes (BacterialSpot, Healthy)
- H: Potato, 3 classes (EarlyBlight, LateBlight, Healthy)
- I: Strawberry, 2 classes (LeafScorch, Healthy)
- J: Tomato, 10 classes (Bacterial_Spot, Early_Blight, Healthy, Late Blight, Leaf Curl Virus, Leaf Mold, LeafSpot, SpiderMite, TargetSpot, Mosaic Virus)

### Table 4 — Plant Disease Classification Performance Metrics (page 13, `papers/10 - PDLCViT_Experiment/inputs/figures/page13.png`)

Visually confirmed full table with exact values:

| Classification Type | Accuracy | Sensitivity | Specificity | Precision | F1 Score | AUC    |
|--------------------|----------|-------------|-------------|-----------|----------|--------|
| A                  | 0.9997   | 0.9996      | 0.9998      | 0.9995    | 0.9995   | 0.9993 |
| B                  | 0.9909   | 0.9907      | 0.9968      | 0.9904    | 0.9906   | 0.9904 |
| C                  | 0.9958   | 0.9941      | 0.9971      | 0.9965    | 0.9953   | 0.9940 |
| D                  | 0.9922   | 0.9921      | 0.9974      | 0.9907    | 0.9914   | 0.9903 |
| E                  | 0.9902   | 0.9863      | 0.9965      | 0.9902    | 0.9882   | 0.9860 |
| F                  | 0.9966   | 0.9861      | 0.9983      | 0.9889    | 0.9875   | 0.9860 |
| G                  | 0.9972   | 0.9970      | 0.9973      | 0.9960    | 0.9965   | 0.9961 |
| H                  | 0.9898   | 0.9871      | 0.9949      | 0.9750    | 0.9809   | 0.9821 |
| I                  | 0.9987   | 1.0         | 0.9982      | 0.9956    | 0.9978   | 0.9971 |
| J                  | 0.9858   | 0.9781      | 0.9984      | 0.9788    | 0.9784   | 0.9781 |

### Table 6 — Performance Metrics Under Ablation Study (page 18, `papers/10 - PDLCViT_Experiment/inputs/figures/page18.png`)

Visually confirmed full table:

| Model                                     | Binary Acc | Binary Sn | Binary Sp | Binary Pre | Binary F1 | Multi Acc | Multi Sn | Multi Sp | Multi Pre | Multi F1 |
|------------------------------------------|-----------|-----------|-----------|------------|-----------|-----------|----------|----------|-----------|----------|
| PDLC-ViT (Baseline)                       | 0.9997    | 0.9996    | 0.9998    | 0.9995     | 0.9995    | 0.9937    | 0.9911   | 0.9975   | 0.9898    | 0.9904   |
| PDLC-ViT without co-attention & co-scale  | 0.9497    | 0.9496    | 0.9498    | 0.9495     | 0.9495    | 0.9241    | 0.9217   | 0.9276   | 0.9205    | 0.9211   |
| PDLC-ViT without Segmentation Head        | 0.8897    | 0.8896    | 0.8898    | 0.8896     | 0.8896    | 0.8546    | 0.8524   | 0.8578   | 0.8512    | 0.8517   |

---

## IMPLEMENTATION NOTES FOR REPRODUCTION

**Framework:** MATLAB 2023a with Deep Learning Toolbox and Image Processing Toolbox
(NOTE: The paper uses MATLAB, NOT Python/PyTorch/TensorFlow)

**Key Architecture Components:**
1. Image Splitter: Divides input image into non-overlapping patches X = {x1, ..., xn}
2. Co-scale Mechanism: Captures multi-scale dependencies between patches → CS
3. Co-attention Mechanism: Captures inter-patch dependencies → CA
4. Cross-attention Module: Two parallel MHSA branches (one for CS, one for CA) → Z1, Z2 → MHCA → Z (shared)
5. Transformer Encoder: Linear Projection → Positional Encoding → Multi-Head Self-Attention → Add&Norm → FeedForward → Add&Norm → Encoded Features
6. Segmentation Head: 3 blocks with residual connections: [LN+MHSA] → [LN+MHCA(+Z)] → [LN+MLP(ReLU)] → Predicted Mask
7. Classification Head: 2 blocks: [LN+MHCA(+Z)] → [LN+MLP(Sigmoid)] → Predicted Labels

**Loss Function:**
- L_total = λ_seg × L_seg + λ_cls × L_cls
- L_seg: Segmentation loss (pixel-wise)
- L_cls: Categorical cross-entropy classification loss
- λ values: Not explicitly specified (grid-searched)

**Training Setup:**
- Optimizer: Adam
- Initial LR: 0.001
- LR Schedule: ×0.5 every 15 epochs
- Early stopping: validation loss threshold = 0.001
- Batch size: 16
- Dropout: 0.2 (encoder and decoder)
- L2 regularization: 0.0005
- Data split: Training vs Testing split shown in Table 1 (54,306 training + 54,306 testing — NOTE: same numbers suggest the table shows per-class counts used in both phases, not a 50/50 split; the PlantVillage dataset is used for classification, PlantDoc (2940 images) for localization/segmentation evaluation)
- No fixed epoch count; early stopping controls training duration

**Datasets:**
- PlantVillage: 54,306 images, 38 classes, used for classification training+evaluation
  - Data augmented with affine transformations
- PlantDoc: 2940 images, 13 species, with segmentation masks, used for localization evaluation

**Hardware:** Intel Core i7, 16 GB DDR4 RAM, NVIDIA GeForce RTX 3070 8 GB
