# PAPER CONTENT: Real-Time Plant Disease Dataset Development and Detection of Plant Disease Using Deep Learning

**Source PDF:** papers/04- Real-Time Plant Disease Dataset Development/inputs/Real-Time_Plant_Disease_Dataset_Development_and_Detection_of_Plant_Disease_Using_Deep_Learning.pdf
**Extracted:** 2026-06-16

---

Received 14 December 2023, accepted 19 January 2024, date of publication 25 January 2024, date of current version 2 February 2024.
Digital Object Identifier 10.1109/ACCESS.2024.3358333

Real-Time Plant Disease Dataset Development
and Detection of Plant Disease Using
Deep Learning

DIANA SUSAN JOSEPH 1, (Student Member, IEEE), PRANAV M. PAWAR                                 1, (Member, IEEE),
AND KAUSTUBH CHAKRADEO 2

1Birla Institute of Technology and Science Pilani, Dubai Campus, Dubai, United Arab Emirates
2Department of Public Health, University of Copenhagen, 1356 Copenhagen, Denmark

Corresponding author: Pranav M. Pawar (pranav@dubai.bits-pilani.ac.in)

ABSTRACT Agriculture plays a significant role in meeting food needs and providing food security for
the increasingly growing global population, which has increased by 0.88% since 2022. Plant diseases can
reduce food production and affect food security. Worldwide crop loss due to plant disease is estimated to be
around 14.1%. The lack of proper identification of plant disease at the early stages of infection can result
in inappropriate disease control measures. Therefore, the automatic identification and diagnosis of plant
diseases are highly recommended. Lack of availability of large amounts of data that are not processed to a
large extent is one of the main challenges in plant disease diagnosis. In the current manuscript, we developed
datasets for food grains specifically for rice, wheat, and maize to address the identified challenges. The
developed datasets consider the common diseases (two bacterial diseases and two fungal diseases of rice,
four fungal diseases of maize, and four fungal diseases of wheat) that affect crop yields and cause damage
to the whole plant. The datasets developed were applied to eight fine-tuned deep learning models with the
same training hyperparameters. The experimental results based on eight fine-tuned deep learning models
show that, while recognizing maize leaf diseases, the models Xception and MobileNet performed best
with a testing accuracy of 0.9580 and 0.9464 respectively. Similarly, while recognizing the wheat leaf
diseases, the models MobileNetV2 and MobileNet performed best with a testing accuracy of 0.9632 and
0.9628 respectively. The Xception and Inception V3 models performed best, with a testing accuracy of
0.9728 and 0.9620, respectively, for recognizing rice leaf diseases. The research also proposes a new
convolutional neural network (CNN) model trained from scratch on all three food grain datasets developed.
The proposed model performs well and shows a testing accuracy of 0.9704, 0.9706, and 0.9808 respectively
on the maize, rice, and wheat datasets.

INDEX TERMS Deep learning, convolutional neural network, dataset development, plant disease
classification, transfer learning.

I. INTRODUCTION                                                         automatic recognition of plant diseases by image analysis is
Plant diseases are among the leading causes of crop loss in             essential to solving this problem. Estimating the disease's
many countries. Traditional disease analysis methods involve            severity is equally important to predict yield and suggest
visual assessment by experts [1], [2]. However, it takes longer         treatment [4].
to diagnose the disease compared to automation techniques.
Experts may only be available in some countries [3]. So,                   Rice, wheat, and maize are important food grains with
                                                                        many health benefits. They are grown in large quantities
   The associate editor coordinating the review of this manuscript and  and are energy giving food to the world. Food grains are
approving it for publication was Miaohui Wang.                          an inevitable source of food for the growing population
                                                                        in many countries. These can be affected by a variety of

16310  
 2024 The Authors. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License.      VOLUME 12, 2024
                                     For more information, see https://creativecommons.org/licenses/by-nc-nd/4.0/
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

diseases and pests. The diseases affecting them can further         less number of layers, and the experimental results show that
prevent the growth of the crop or affect the yield. Among           it generalizes well on the datasets. As a future enhancement,
the different food grains, the diseases affecting the rice,         the model can be experimented with for disease diagnosis of
wheat, and maize/corn plants are given importance in this           other plant leaf datasets.
manuscript. Some common diseases affecting rice plants are
Rice Blast, Sheath Blight, Bacterial Leaf Streak, Bacterial            The main contributions of the current research work are as
Leaf Blight, etc. [5]. The wheat plants' diseases include           follows:
Powdery Mildew, Septoria Leaf Spot, Tanspot, Snowmold,
etc. [6]. Charcoal rot, Downey mildew, Commonrust, black               1) A real-time dataset for three important food grains,
bundle disease, etc., are some diseases that commonly affect               rice, wheat, and maize, was developed. Four diseases
maize plants.                                                              affecting each crop and a healthy class are present in
                                                                           all three datasets.
   CNN [7], [8], [9] is a popular method that has been widely
used in the field of plant disease detection [10], [11], [12]. The     2) Early, advancing, and severe stages of the diseases were
dataset used by the CNN models should be of good quality,                  focused on while developing the datasets. Then, all
and it has to be properly labeled. The performance of the CNN              three stages of a particular disease were put together
models greatly depends on the quality and amount of data                   in one class.
used to train the model. The deep learning based CNN models
require a large amount of data for training. It will not perform       3) A hundred images were collected under each class for
well if the model is trained using less data.                              all five classes in a dataset. The collected images are
                                                                           pre-processed if needed and augmented to increase the
   A dataset is an important component in machine learning                 size of the dataset.
or deep learning to learn and evaluate the performance of
any algorithm or model. A dataset helps to organize the                4) A hundred images under each class of the datasets col-
information collected from different sources and to arrive                 lected were annotated manually using the MakeSense
at the desired outcome. Only a well-organized dataset used                 AI tool for use in object detection models like Mask
for training a machine learning or deep learning model                     R-CNN.
will be able to reflect the quality of the model and its
effectiveness in arriving at the target outcome. One of the            5) Eight CNN models were fine-tuned, trained, and tested
main challenges in plant disease diagnosis is the lack of                  on the datasets developed. The performance of the
availability of large and non-lab datasets. Some of the                    fine-tuned CNN models and the proposed CNN model
disadvantages of the existing datasets are most of the datasets            on the developed real-time datasets was evaluated using
available are laboratory processed; they include mostly                    the accuracy and loss performance graphs, confusion
images taken under controlled settings, a single leaf is                   matrices, classification reports, etc., on the test data.
mostly considered, they don't include all the symptoms of
a particular disease, and complex backgrounds with varying             6) A CNN model, namely MRW-CNN, is proposed
lighting conditions are not considered. These disadvantages                for detecting maize, rice, and wheat leaf diseases.
can limit the performance of disease detection by a CNN                    The proposed model can identify the features of the
model, as the images can have multiple leaves with different               diseased and healthy maize, rice, and wheat leaves and
complex backgrounds.                                                       accurately detect the diseases.

   Our work focussed on these challenges, and three datasets           7) As a future extension of the current work, the different
for three important food grains, rice, wheat, and maize,                   stages of the diseases included under a particular
were developed. The images in the datasets included all the                disease can be separated into classes like early,
disease stages early, advancing, and severe of all the diseases            advancing, and severe, and predictions on disease
considered. The datasets includes processed images, images                 stages can be made.
under complex backgrounds and lighting conditions, and
images with multiple leaves under complex backgrounds.              In the current work during the recognition of the maize leaf
The images in the datasets were collected from internet             diseases, the models MobileNet, MobileNetV2, Inception
sources and existing datasets. In the current work, the             V3, InceptionResNetV2, and Xception performed well with
detection and classification of the diseases considered in          a testing accuracy of above 90%. For rice leaf disease recog-
the manuscript affecting rice, wheat, and maize were made           nition, all the models except ResNet 50 and ResNet 101 gave
using CNN models. Pre-trained eight CNN models trained              a testing accuracy above 90%. The models MobileNet,
on the ImageNet dataset were fine-tuned, trained, and tested        MobileNetV2, Inception V3, and InceptionResNetV2 gave
on the datasets developed. A CNN model, namely Maize,               a testing accuracy of above 90% on the wheat leaf dataset.
Rice, Wheat-CNN model (MRW-CNN), is also proposed                   The experimental results demonstrate that when trained from
in our work to evaluate the performance of the developed            scratch on all the datasets developed, the proposed model
datasets. Compared to the existing heavy state-of-the-art           gave a testing accuracy above 95%. The proposed model
models for plant disease diagnosis, the MRW-CNN model has           proved to generalize well on all three datasets considered in
                                                                    our work.
VOLUME 12, 2024
                                                                       The rest of the paper is organized into seven sections.
                                                                    Section II deals with the related works in plant disease
                                                                    diagnosis. It focuses on the literature review of the works
                                                                    concentrating on dataset development for different plants and
                                                                    CNN models used for plant disease diagnosis. Section III

                                                                                                                                                                   16311
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

deals with the steps in data set development, like data           conditions. Several experiments were conducted using the
collection, pre-processing, augmentation, and annotation.         CNN models VGG19, ResNet152, DenseNet169, Inception-
Section IV discusses the development of the fine-tuned            NetV3, MobileNetV2, and eight variants of EfficientNet
models, like how the CNN models selected are fine-tuned and       architecture; the results showed the fine-tuned EfficientNet
trained, the basic architecture used, etc. Section V deals with   B4 model proposed in the study performed better with a
the development of the proposed MRW model. Section VI             testing accuracy of 99.35% when compared to the other
deals with the results of the experiments conducted on the        models.
CNN models with the datasets developed for each food grain,
experimental setup, hyperparameters used, visualization of           Leaf tip detection and the number of leaves in a rice
activations, etc. Section VII discusses the case study of the     plant were determined by training a You Only Look Once
application of the proposed method. Finally, Section VIII         (YOLO) deep learning algorithm in [23]. The study in [24]
deals with the conclusion and future work.                        proposed a model that uses machine learning and deep
                                                                  learning advantages. The proposed model includes forty
II. RELATED WORKS                                                 different hybrid deep learning models and five machine-
The review conducted in [13] mainly focuses on current            learning techniques. The hyperparameters of the classifiers
developments in plant disease diagnosis using techniques like     were optimized using the Optuna framework. A dataset
machine learning and deep learning and how these techniques       for tomato early blight disease was collected from the
speed up the process of disease detection in plants. The study    Indian Agricultural Research Institute. The proposed model
elaborates on the importance of the detection of pests and        performed well on the dataset (IARI-TomEBD) formed
diseases in plants, the various datasets used in plant disease    and gave an accuracy of 87.55­100%. Validation of the
detection, performance metrics used to evaluate the plant         model was also performed on PlantVillage-TomEBD and
disease detection models, the challenges in the field, and        PlantVillage-BBLS.
future enhancements.
                                                                     A deep learning approach was proposed in [25] to identify
   A 3DCNN is built to identify plant diseases in [14]; the       diseases affecting maize crops. A dataset was prepared
work focussed on lesion segmentation and plant survival           from fields of ICAR-IIMR, Ludhiana, India. Three different
probability. The proposed model was evaluated on the              architectures based on the Inception-v3 framework were
diseases affecting three plants, pepper, potato, and tomato,      trained on the dataset prepared. The best performing model
from the Plant Village dataset. The model achieved a DICE         obtained an accuracy of 95.99%. A comparison of the state-
coefficient of 90% and classification accuracies of 91.11%,       of-the-art models with the model that performed best was
93.01%, and 99.04% for pepper, potato, and tomato plants,         done.
respectively.
                                                                     Analysis of Rice Blast disease by deep learning based
   A lightweight CNN model was developed in [15] based            model is done in [26] using remote sensing images. The
on VGG19 for disease diagnosis in peach leaves. Bacteriosis       proposed model obtained a training accuracy of 90.02% and a
infection is mainly focussed on in the manuscript. The dataset    validation accuracy of 85.33%. Moderate resolution imaging
for conducting the experiments was obtained from Farm             spectroradiometer is used to establish the spatial distribution
of Agriculture University Peshawar, Pakistan. A total of          of leaf blasts. A deep learning based approach was used
10,000 images were present in the dataset, and augmentation       in [27] to detect tomato plant diseases. Two state-of-the-art
techniques were used to increase its size. The images in the      semantic segmentation models, U-Net and Modified U-Net,
dataset were also pre-processed.                                  were used to detect and segment the disease affected regions.
                                                                  Different U-Net versions were looked into for selecting the
   The plant village dataset [16] contains 54,303 healthy         optimum one. Different classification methods were used in
and unhealthy images of leaves. The images in the dataset         the study; both binary and multiclass classification of tomato
are laboratory processed. The plant village dataset is an         diseases were done.
open-access repository of images that can help to detect
and classify plant diseases. The dataset is divided into             A dataset was developed by collecting 500 images of
38 categories by species and diseases. Many studies in the        maize leaves from Google websites and the plant village
field of plant disease diagnosis have made use of the plant       dataset in [28]. The different periods of occurrence of maize
village dataset [17], [18], [19], [20].                           diseases are included in the dataset. The size of the dataset
                                                                  was increased to 3060 images by applying augmentation
   Temperature and carbon dioxide effect on maydis leaf           techniques. There were eight categories of diseased leaves
blight disease affecting maize was analyzed in [21]. The          in the dataset. The study proposed two improved GoogLeNet
results of the study showed that the severity of the disease can  and Cifar10 models for plant disease recognition. Eight kinds
increase with the increase in temperature. Tolerance attributes   of maize diseases were identified.
were calculated for six disease stress based on diseased
and non-diseased according to the yield. Wheat disease               A dataset of maize leaves was developed with the help
identification model based on EfficientNet architecture was       of an Unmanned Aerial Vehicle (UAV) in [29]. The dataset
proposed in [22] for detecting wheat rusts. A dataset             contains 6267 images, where 3741 images are with lesions
consisting of 6556 images was prepared from natural field         and 2526 are without lesions. The maize leaf images in the
                                                                  dataset are field images of the disease Northern Leaf Blight.
16312
                                                                                                                                                                   VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

A CNN model ResNet-34 pre-trained on the ImageNet                 images were annotated using 300 human hours. The images
dataset was used in the study for training. The CNN model         in the dataset were downloaded from Google Images and
trained was used to generate heat maps that showed the            Ecosia. Three models were used in the study to show the
disease regions.                                                  efficacy of the proposed dataset. An increase in classification
                                                                  accuracy by 31% was reported in their work.
   The NLB dataset is used in [30]. It has three different
parts. The first part is hand-held. The second part is the boom      From the literature survey conducted, it can be concluded
set. The last data set is an unmanned drone set. The hand-held    that most of the datasets developed do not cover all of
part has more clarity; thus, the hand-held part of the dataset    the underlying challenges existing in the field of plant
is used in this study. The dataset contains 1019 images           disease diagnosis, like complex backgrounds, the variations
with different angles and backgrounds and 7669 annotations.       of disease symptoms according to climate, the occurrence
Augmentation techniques were applied to the dataset, and the      of multiple diseases on plants, etc. Only a few datasets
size of the dataset was increased to 8152 images. A multi-        are publicly available for further research or study. Some
scale feature fusion instance detection method was proposed,      concentrate on the stages of the different diseases, but mainly,
based on CNN (ResNet-101), to recognize the disease maize         it is limited to one or two plants. Therefore, there is a
leaf blight.                                                      need to develop a dataset that will address most of the
                                                                  underlying challenges in the field. The current manuscript
   A dataset of maize leaves was formed in [31] by collecting     mainly concentrates on developing a dataset of good quality
some images from the corn plantations in Raebareli and            for food grains rice, wheat, and maize. The dataset's primary
Sultanpur district, the rest from the Plant Village dataset. The  focus is on real-life leaf images of the food grains considered.
dataset had three categories of images: Rust, Northern Leaf       For example, Common Rust affecting the maize plants; for
Blight, and Healthy. An Agriculture Scientist from India was      this particular disease, real-life images of all the stages are
consulted to label the images captured. A model was formed        considered early, middle, and advancing and are included
based on DCNN to identify two diseases affecting the maize        under CommonRust disease class, and the same for all
plants.                                                           diseases in the dataset developed.

   The citrus dataset developed in [32] contains 759 images       III. PLANT DISEASE DATASET DEVELOPMENT
of leaves and citrus fruits, which are healthy and diseased.      A real-time plant disease dataset for three important food
The diseases considered in the dataset are black spot, canker,    grains, rice, wheat, and maize, based on leaf images, was
scab, greening, and melanose affecting the citrus plants. The     developed in the current study. Diseases that cause great yield
images in the dataset were manually captured using a DSLR         loss and have similar symptoms, many samples available,
with the help of an expert.                                       etc., were selected to form the dataset. Four diseases were
                                                                  selected for each food grain. For the diseases considered
   A real-time dataset of wheat diseases was developed            under each food grain, while collecting the images from
in [33]. The dataset peers into most challenges for wheat         internet sources, the leaf images for different stages of a
disease diagnosis, specifically complex backgrounds, differ-      particular disease, specifically early, advancing, and severe,
ent capture conditions, etc. A model using deep learning          were collected and put under the same disease class. More
and multiple instances learning for the real-time detection of    details on the dataset developed and the samples collected are
wheat diseases was proposed in the study.                         available in Table 1, Table 2, and Table 3. In the current work,
                                                                  the diseased and healthy leaf images collected for each food
   A dataset of wheat leaves was developed in [34] to detect      grain are pre-processed, augmented, and then applied to deep
the Fusarium Head Blight disease (FHB). The images in             learning based CNN models for detecting and classifying
the dataset were collected from the field. Twelve images          diseases. The results obtained are discussed in Section VI.
were collected initially, and later, it was increased to          Later, for the future extension of the work to be done, the
2829 by applying data augmentation techniques. In this study,     images were manually annotated after the data collection
a method was proposed that uses Mask-RCNN along with              and pre-processing steps for use in object detection models.
color images to detect FHB.                                       Around 1500 images were annotated for all three food grains.

   The rice diseases image database was created with                 The overall process involved in developing the plant
500 common rice disease images of leaves and stems                disease dataset for rice, wheat, and maize food grains in
collected from the experimental field in [35]. This work trains   the current work is shown in Fig. 1. In Fig. 1, the different
the CNNs to identify ten diseases affecting rice plants. A real-  processes involved in developing the datasets are shown. The
time dataset of rice leaves was developed in [36]. The dataset    figure mainly shows the food grains and their stages selected
contains a total of 8911 images. The rice leaf images included    in our work, the sources from which the datasets' images
in the database contain both diseased and healthy images. The     are collected, and the pre-processing and augmentation
diseases considered are Rice blast, Red blight, Stripe blight,    techniques used for processing the collected images. The
and Sheath blight. CNN was used for feature extraction, and       different steps in developing the datasets are detailed in
d support vector machine (SVM) was used for predicting            Subsections A, B, C, and D.
diseases.
                                                                                                                                                                 16313
   PlantDoc, a Visual Plant Disease Detection dataset, was
developed in [37]. The dataset contains 2,598 data points
across 13 plant species. It includes 17 classes of diseases. The

VOLUME 12, 2024
                                                 D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

TABLE 1. Details of maize leaf disease dataset.

TABLE 2. Details of rice leaf disease dataset.
TABLE 3. Details of wheat leaf disease dataset.

A. DATA COLLECTION                                           manually selected and filtered. The dataset also comprises a
The dataset includes real-time images collected from Google  few processed images for the healthy maize, rice, and wheat
websites, Ecosia, Bing images, Flickr, etc. The images were  classes. Few images were taken from the plant village [16]

16314                                                                                                                                                         VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

FIGURE 1. Dataset development methodology.

dataset for the healthy class of maize, and the healthy class      for maize, rice, and wheat plants. The whole process of data
of rice and wheat images were downloaded from the Kaggle           collection is shown in Fig. 2. The samples of a few raw images
website [38], [39]. The dataset was created by downloading         collected for each food grain rice, wheat, and maize for the
images from the internet. A total of 5 classes were considered     development of the datasets are shown in Fig. 3, Fig. 4, and
for each plant, including the healthy class, a different class     Fig. 5.
from the diseased class. For all the diseased classes, images
for all the stages of the disease were attempted to be collected,  B. DATA PRE-PROCESSING
like images of early infection, advancing of the disease stage,    Data pre-processing helps enhance some features of the
and finally, the severe stage.                                     image data required for processing. It can help in improving
                                                                   the accuracy and reliability of a dataset. Some images down-
   CNN models trained on real-life images are believed             loaded from the internet for the datasets were pre-processed
to perform better when compared to laboratory processed            before training the model. The software tools used for
images. Therefore real-life images were given more impor-          pre-processing the images were ImageJ [67], [68], and
tance while developing the dataset when compared to                Python code. The purpose of pre-processing was to remove
processed images. The crops considered for developing the          foreign objects from the images. The pre-processing steps
dataset are rice, wheat, and maize. Four diseases affecting        applied to some images were resize, crop, sharpen, contrast
each of these plants were assessed for developing the dataset.     enhancement, brightness and contrast adjustment, gaussian
The diseases considered for rice plants are Rice Bacterial         blur, etc. Python code was used to resize all the images in
Blight, Rice Bacterial Leaf Streak, Rice Blast, and Rice           the dataset. The images were resized to 224 × 224. Some
Brown Spot.                                                        images were cropped using the ImageJ software and Python
                                                                   code to select the diseased areas of the leaves other than the
   Similarly, the diseases considered for the wheat plant are      background.
Leaf Rust, Powdery Mildew, Stripe Rust, and Tan Spot. The
diseases looked into for maize plants are Common Rust, Grey           Some images in the dataset were manually selected and
Leaf Spot, Northern Leaf Blight, and Southern Rust. Table 1,       sharpened to visually focus on the diseased areas of the leaves
Table 2, and Table 3 give detailed descriptions regarding          more clearly. Similarly, other pre-processing techniques like
the type of disease, the number of images collected under          contrast enhancement of the images to increase or decrease
each class and the number of images collected for early,           the contrast, brightness adjustments to increase or decrease
advancing, and severe stages of the disease in each class, the     the brightness of specific images, and gaussian blur to
areas of the leaves affected, and the conditions favorable for     blur certain portions of the image were done using the
the disease to occur and advance; for the datasets developed
                                                                                                                                                                  16315
VOLUME 12, 2024
                            D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

FIGURE 2. Data collection.

FIGURE 3. Samples of a few raw diseased images used to form the rice leaf dataset [40], [41].

ImageJ software. Examples of a few pre-processed images           Python code was used to augment all the images in the
of Common Rust disease affecting the maize plant from the         dataset. Offline augmentation was done to all the images and
datasets are shown in Fig. 6.                                     saved to a folder. Then, the original images were combined
                                                                  with the augmented images of each class to form the dataset.
C. DATA AUGMENTATION
A total of hundred images were collected under each class            The images in all three datasets collected were augmented
for the plants considered in this manuscript. The number          by applying specific augmentation techniques like horizontal
of images under each class was insufficient to train a deep       and vertical flipping of the images, rotating the images by
learning model. Therefore, data augmentation techniques           90 degrees, applying a height shift range of 0.2, applying
were applied to the dataset to increase the size of the dataset.  a width shift range of 0.2, zooming and shearing of the
                                                                  images by a range of 0.2, a brightness range of 0.2 to 0.8 was
16316
                                                                                                                                                                   VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

FIGURE 4. Samples of a few raw diseased images used to form the wheat leaf dataset [42], [43], [44], [45], [47], [48], [49], [50], [51], [52], [53].

FIGURE 5. Samples of a few raw diseased images used to form the maize leaf dataset [54], [55], [56], [57], [59], [60], [61], [62], [63], [64], [65], [66].

applied to the images. Examples of a few augmented images    labels to be assigned to the images are mentioned. Each image
of Tanspot and Powdery Mildew diseases from the datasets     uploaded is selected individually, and the polygon bounding
are shown in Fig. 6. Data augmentation techniques helped     box is selected to annotate the images. After annotating
increase the model's ability to generalize and reduced the   all the images uploaded to the web browser, the labels
problem of overfitting.                                      can be exported in YOLO, VOC, XML, VGG, JSON, and
                                                             CSV formats. Since the polygon bounding box was used
D. DATA ANNOTATION                                           for annotation, the labels were exported in VGG and JSON
Data annotation can help to understand the meaning of        formats.
data more precisely, increasing the performance of the deep
learning models. The Make Sense AI [69] Tool was used        IV. DEVELOPMENT OF FINE-TUNED MODELS
in the current work to annotate the images in the dataset       The collected, pre-processed, and augmented datasets for
under each class. The Make Sense AI is an open-source
and free to use annotation tool. It runs on a web browser.   each food grain are split 80% for training and 10% each
The different stages of a particular disease in the dataset  for validation and testing. The datasets for each food grain
were given importance during labeling. The whole process     had a total of around 5000 images under each class after
of data annotation is shown in Fig. 7. In the figure, the    augmentation. The size of all three datasets was increased
images of Common Rust disease affecting the maize plants     from 1500 images to 25000 images after augmentation. After
are loaded into the Make Sense AI Tools web browser. The     splitting the dataset into training, validation, and testing,
                                                             each class contains 4000 images for training and 500 images
VOLUME 12, 2024
                                                                                                                                                            16317
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

                   FIGURE 6. Data pre-processing and data augmentation.  deep learning models like VGG16, ResNet 50, ResNet
                                                                         101, Inception V3, InceptionResNetV2, MobileNetV2, and
Algorithm 1 Fine-Tuning of CNN Models                                    MobileNet. Pre-trained models trained on the ImageNet
Input: Pre-trained models with top layers removed                        dataset [70] were downloaded from Keras and used. The
Output: A fine-tuned model for detection and classification              concepts of transfer learning and fine-tuning were made use
                                                                         of for the training. Algorithm 1 discusses how the pre-trained
     of diseases from the dataset developed                              models trained on the ImageNet dataset were fine-tuned.
 1: Load pre-trained models for fine-tuning                              The steps in the algorithm include the following: the models
 2: m  loadedmodel(pre - trained = True)                                 were downloaded from Keras. The top layers from the eight
 3: while m.toplayer = False do                                          models downloaded were removed. The layers trainable
 4: Set layers.trainable = False                                         is set to False. After which, a few layers like Dropout,
 5: x  dropout(0.2)(m.output)                                            BachNormalization, Flatten, Dense layer, and the Softmax
 6: x  addBachNormalization(x) {x is output from                         layer for the output classes were added.

        previous layer}                                                     A Dropout layer was added to nullify the effect of some
 7: x  addFlattenlayer(x)                                                neurons on the next layer. The BachNormalization layer
 8: x  addDenselayer(x)                                                  works differently during training and inference. The layer can
 9: l2regulizer  0.01                                                    help the model to generalize well by normalizing its inputs in
10: x  dropout(0.2)(x)                                                   batches. The Bach Normalization layer can help smoothen
11: predictions  outputclasses(activation = softmax)                     the loss function by optimizing the model parameters and
12: finetunedmodel  (m.input, predictions)                               increasing the models' training speed. A Flatten layer was
13: end while                                                            added to convert the inputs to one-dimensional. The Dense
                                                                         layer was added after the Flatten layer with a kernel-
each for testing and validation. The number of images in                 regularizer l2 of 0.01. Again, a Dropout of 0.2 was added.
all classes was balanced in all three datasets. The dataset
developed for each plant was applied to various CNN based                                                                                                                 VOLUME 12, 2024

16318
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

                             FIGURE 7. Example of annotation of commonrust disease stages and healthy stage of maize using the  16319
                             makesense AI tool.

VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

The Softmax layer was added to get the output classes.            CNN model was done by splitting the dataset developed for
Finally, a fine-tuned model was obtained where the input          the three food grains into training, validation, and testing
is the model's input, and the output is the predictions.          datasets. The datasets were split into 80% for training and
After conducting several experiments, a Dropout of 0.2 and        10% each for validation and testing. In the training phase, the
kernel-regularizer l2 and Bach Normalization layer, Early         input images from the training set that are pre-processed and
stopping was decided to be added to the pre-trained models to     augmented are given to the proposed model for training. The
prevent over-fitting. Early Stopping automatically terminates     model is then validated on the validation dataset. In the testing
the training when a performance measure does not improve          phase, the trained MRW-CNN model is tested on the testing
further. The model architecture used is shown in Fig. 8.          data which the model has not seen during training, predictions
                                                                  are then made. The training, validation, and testing datasets
   Fig. 8 contains mainly two phases: the training and            contain samples from all classes in the dataset. A total of five
testing phases. In the training phase, the input images           output classes are available for each of the datasets developed
from either of the three datasets, which are pre-processed        in our work.
and augmented, are given as input to the state-of-the-art
deep learning models. The datasets are split into 80% for         VI. RESULTS
training, 10% for validation, and 10% for testing before being    A. EXPERIMENT SETUP
given to the CNN models. The pre-trained models ResNet            Eight pre-trained models trained on the Imagenet dataset
50, ResNet 101, MobileNetV2, MobileNet, Inception V3,             were downloaded from Keras and fine-tuned. GPU (NVIDIA
InceptionResNetV2, VGG 16, and Xception, trained on the           ®Quadro ®P1000,4GB,4mDP) was used to train all fine-
ImageNet dataset, are selected in the current work for training   tuned models. Tensorflow, Keras, CUDNN, etc., were used
on the datasets developed. The pre-trained CNN models             for software implementation. The proposed MRW-CNN
selected are fine-tuned by removing the top layers and adding     model was trained using the T4 GPU available in Google
certain layers, as mentioned in Algorithm 2. The CNN models       CoLab.
fine-tuned are trained on the training data and validated on the
validation data. Examples of feature maps from a few layers       B. HYPERPARAMETERS
of the InceptionResNetV2 model after training on the input        All the fine-tuned CNN based deep learning models are
images are also shown in Fig. 8.                                  optimized by the SGD algorithm. The learning rate for all
                                                                  the models was set to 0.001, and a batch size 32 was chosen.
   In the testing phase, the trained CNN models are tested        Dropout operation was added to all the pre-trained models
on the testing data, and predictions are made; the testing        on the ImageNet dataset to prevent over-fitting and improve
data is the data the model has not seen before. Each dataset      the generalization of the models. The MRW-CNN model is
developed for the food grains rice, wheat, and maize has five     optimized by the SGD, a learning rate of 0.001 and batch size
classes each. Therefore, a total of 15 output classes are shown   of 32 were used.
in Fig. 8.
                                                                  C. CONFUSION MATRIX
V. PROPOSED CNN MODEL OF MAIZE, RICE AND WHEAT                    A confusion matrix can greatly help in visually estimating the
(MRW-CNN) LEAF DISEASES                                           performance of a model. The infected images of food grains
Fig. 9 shows the architecture of the proposed CNN model that      considered in the current manuscript can be easily confused
has been used to classify the diseases affecting the food grains  with multiple classes since most of the diseases selected for
considered in our manuscript. The model consists of five          the development of the dataset have similar symptoms. The
Convolution layers, each followed by BatchNormalization           infected leaf images at different backgrounds can lead to
and Rectified Linear Unit (ReLU). After the convolution           confusion and complexity, which can result in performance
layers, a Global Average Pooling layer is added, followed by a    deterioration.
Dense layer with a Softmax activation function that acts as the   The confusion matrix [71] is an m*m matrix that can be used
output layer. The model is trained using Stochastic gradient      to measure the performance of the classification models. In a
descent (SGD). In the proposed CNN model, the convolution         confusion matrix, the X-axis contains the predicted values,
layers were used to extract the features from the training        and Y-axis includes the actual values. The True Positive
images; batch normalization is used to normalize the outputs      (TP), True Negative (TN), False Positive (FP), and False
from each layer during training for every batch. The layer        Negative (FN) for each class in a multi-class classification
helps to address the internal covariate shift problem. The        can be computed from the confusion matrix, and from
activation function ReLU does not activate all the neurons at     these, the values the precision, recall and F1-Score can
the same time; this helps to make the computation easier.         be calculated per-class. The precision measures out of all
                                                                  predicted positives how many are actually positive. It can be
   The problem of overfitting can be reduced by using the
Global Average Pooling layer. The images can be down                                                                                                               VOLUME 12, 2024
sampled using average pooling. The Dense layer is used for
predicting the output class of the input image. The output of
the Dense layer is used to evaluate the performance of the
proposed model. The training and testing of the proposed

16320
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

                 FIGURE 8. Architecture of Fine-tuned CNN Models.                                16321
VOLUME 12, 2024
                                                       D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

FIGURE 9. Architecture of proposed CNN model MRW-CNN.

calculated per-class using the equation (1)                      expressed in equation (6)

       Precision =   TP                                (1)       PrecisionMacroAvg = n Precisioni            (6)

                     TP + FP                                                                   i=1  n

Recall measures how many positive records are predicted          Macro average F1 scores are calculated by taking the
correctly. It can be calculated per-class using equation (2)     arithmetic mean of all the per-class F1 scores as expressed
                                                                 in equation (7)
       Recall =      TP                                (2)

                     TP + FN                                     F1 - scoreMacroAvg = n F1 - scorei

F1-Score is the harmonic mean of precision and recall it can                                        n        (7)
be calculated per-class as expressed in equation (3)
                                                                                               i=1

       F1 - score =  2 × Precision × Recall            (3)       The weighted-averaged F1-score [73] is calculated using
                                                                 equation (8) and equation (9) by taking the mean of all per-
                     Precision + Recall                          class F1 scores for a N-class dataset, while considering each
                                                                 class's support.
D. CLASSIFICATION REPORT
The classification report is a performance evaluation metric                                N
that displays precision, recall,f1-score, accuracy, macro avg,
weighted avg, and support for each class. It is used to measure  Weighted F1 - score = Wi × F1 - scorei (8)
the performance of the trained models on the test data. The
accuracy is calculated by dividing the correct predictions                                  i=1
by the total predictions from the confusion matrix obtained,
as expressed in equation (4)                                     where, Wi  =  No of samples in each classi  (9)
                                                                                 Total number of samples

                                                                 In a similar way, the weighted average scores for precision
                                                                 and recall can also be computed.

       Accuracy =    CorrectPredictions                (4)

                     TotalPredictions                            E. VISUALIZATION OF ACTIVATIONS

Macro average recall scores are calculated by taking the         In some cases, the models during training are trained on some
arithmetic mean of all the per-class recall scores [72] as       background or irrelevant features, which gives a satisfactory
expressed in equation (5).                                       model accuracy. Such situations can be checked by generating
                                                                 activation maps. The activation maps can help determine if
       RecallMacroAvg = n Recalli                      (5)       the models focus on the right features. For example, the
                              n                                  output of the visualization of activations created by Grad-
                     i=1                                         cam [74] for the MobileNet model given three input images
                                                                 from the maize leaf dataset is as shown in Fig.10. The figure
Macro average precision scores are calculated by taking          shows that the model focuses on the suitable feature for all the
the arithmetic mean of all the per-class precision scores as

16322                                                                                                  VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

                                                                       FIGURE 12. Accuracy and loss performance of the model MobileNet on
                                                                       the maize leaf dataset.

FIGURE 10. Activation maps generated by Grad-CAM given three input
images (a) input images (b) activation maps generated from last
convolutional layer.

FIGURE 11. Accuracy and loss performance of the model xception on the  FIGURE 13. Normalized confusion matrix of the model xception on the
maize leaf dataset.                                                    maize leaf dataset.

input images from the maize leaf dataset, the images from              a validation accuracy of 0.9336 and a testing accuracy of
three different classes CommonRust, GreyLeafSpot, and sr               0.9312. The fine-tuned Xeception model performed best
were given as input to the trained MobileNet model.                    with a testing accuracy of 0.9580. The graphs showing the
                                                                       accuracy and loss performance of the models Xception and
F. MAIZE                                                               MobileNet showing better testing accuracy when compared
The performance of different models on the maize dataset               to other fine-tuned models on the maize leaf dataset are shown
is shown in Table 4. The maize dataset includes five                   in Fig. 11 and Fig. 12. The normalized confusion matrix of the
classes, including the healthy class, for analysis. Xception,          models Xception and MobileNet on the maize leaf dataset is
MobileNet, MobileNetV2, Inception V3, and InceptionRes-                shown in Fig. 13 and Fig. 14. The performance of the models
NetV2, models trained on the maize leaf dataset, showed                can be evaluated from the normalized confusion matrix by
better performance when compared to the other fine-tuned               observing the diagonal elements. By referring to the color bar,
models. Xception showed a validation accuracy of 0.9608 and            the darker the color blue of the diagonal elements, the better
a testing accuracy of 0.9580. The InceptionV3 model showed             the performance of the model.
a validation accuracy of 0.9468 and a testing accuracy
of 0.9448. MobileNet showed a validation accuracy of                      In Fig. 13, for the Xception model on the maize leaf
0.9484 and a testing accuracy of 0.9464. The MobileNetV2               dataset, when observing the diagonal values, the Mhealthy
model showed a validation accuracy of 0.9412 and a testing             gives a better value when compared to all the other classes.
accuracy of 0.9404. The InceptionResNetV2 model showed                 Confusion among classes can also be observed. For example,

VOLUME 12, 2024                                                                                                                                                       16323
                                                                                       D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease
TABLE 4. Application of state-of-the-art models on the maize leaf dataset.

                                                                                                                 TABLE 6. Classification report of mobilenet model on the maize leaf
                                                                                                                 dataset.

FIGURE 14. Normalized confusion matrix of the model MobileNet on the
maize leaf dataset.

TABLE 5. Classification report of xception model on the maize leaf
dataset.

                                                                      FIGURE 15. Normalized confusion matrix of the model MRW-CNN on the
                                                                      maize leaf dataset.

CommonRust and sr, nlb and GreyLeafSpot, nlb, Mhealthy,               can also be observed; for example, GreyLeafSpot is highly
etc. In Fig. 14 for the MobileNet model on the maize leaf             confused with the class Mhealthy. There are also confusions
dataset, it can be observed that the diagonal values for all          between the classes sr and CommonRust and also sr and
classes are nearly the same except for one. Therefore, it can be      GreyLeafSpot, etc.
assumed that all classes perform equally the same except for
the disease SouthernRust (sr). Some confusion among classes              The proposed MRW-CNN model trained from scratch on
                                                                      the maize leaf dataset performed well; it showed a validation
16324
                                                                                                                                                                       VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease
TABLE 7. Classification report of MRW-CNN model on the maize leaf
dataset.

                                                                                                                 FIGURE 17. Accuracy and loss performance of the model inception V3 on
                                                                                                                 the rice leaf dataset.

FIGURE 16. Accuracy and loss performance of the model MRW-CNN on  FIGURE 18. Accuracy and loss performance of the model xception on the
the maize leaf dataset.                                           rice leaf dataset.

accuracy of 0.9656 and a 2% increase in testing accuracy of       FIGURE 19. Normalized confusion matrix of the model inception V3 on
0.9704 compared to the fine-tuned models. The normalized          the rice leaf dataset.
confusion matrix of the MRW-CNN model on the maize leaf
dataset is shown in Fig. 15. In Fig. 15, when observing the       showed validation and testing accuracy above 95% when
diagonal elements, the Mhealthy and nlb classes of maize          compared to all the other models trained. The Xception
performed equally well when compared to the other classes.        model performed best when compared to all the other models.
Confusion among classes can also be observed; for example,
the class nlb is highly confused with Mhealthy, the Mhealthy                                                                                                     16325
class is confused with CommonRust, etc. The accuracy and
loss performance of the MRW-CNN model on the maize leaf
dataset is shown in Fig. 16.

   The classification report of the Xception, MobileNet and
the proposed MRW-CNN model on the maize leaf dataset is
shown in Table 5, Table 6 and Table 7. The tables show the
precision, recall, F1-score, and support scores for each class
in the maize leaf dataset on the test data. The macro avg,
micro average, and accuracy are also calculated. More details
regarding how these values can be calculated in mentioned in
Subsection D of Section VI.

G. RICE
The performance of the state-of-the-art deep learning models
on the rice leaf dataset is shown in Table 8. Out of all eight
fine-tuned models trained on the rice leaf dataset, the models
Xception, MobileNet, Inception V3, and MobileNetV2

VOLUME 12, 2024
                                                                                       D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease
TABLE 8. Application of the state-of-the-art models on the rice leaf dataset.

                                                                                                                 TABLE 10. Classification report of inception v3 model on the rice leaf
                                                                                                                 dataset.

FIGURE 20. Normalized confusion matrix of the model xception on the
rice leaf dataset.

TABLE 9. Classification report of xception model on the rice leaf dataset.

The MobileNetV2 model showed a validation accuracy of                       FIGURE 21. Normalized confusion matrix of the model MRW-CNN on the
0.9624 and a testing accuracy of 0.9584. The MobileNet                      rice leaf dataset.
model showed a validation accuracy of 0.9676 and a testing
accuracy of 0.9572.                                                         InceptionResNetV2 showed a validation accuracy of
                                                                            0.9492 and a testing accuracy of 0.9532. The VGG16 model
   The Inception V3 model showed a validation accu-                         showed a validation accuracy of 0.9148 and a testing accuracy
racy of 0.9628 and a testing accuracy of 0.9620. The                        of 0.9120. The Xception model showed a validation accuracy
                                                                            of 0.9708 and a testing accuracy of 0.9728. The graphs
16326
                                                                                                                                                                             VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease
TABLE 11. Classification Report of MRW-CNN model on the rice leaf
dataset.

                                                                                                                 FIGURE 23. Accuracy and loss performance of the model mobilenet on
                                                                                                                 the wheat leaf dataset.

FIGURE 22. Accuracy and loss performance of the model MRW-CNN on  FIGURE 24. Accuracy and loss performance of the model mobilenetv2 on
the rice leaf dataset.                                            the wheat leaf dataset.

showing the accuracy and loss performance of the fine-tuned       can also be observed, for example, rblast and bls, riceh and
models trained on the rice leaf dataset with a testing accuracy   ricebs, etc. The accuracy and loss performance of the MRW
of above 96% are shown in Fig. 17 and Fig. 18. The                model on the rice leaf dataset is shown in Fig. 22. The
normalized confusion matrix of the fine-tuned models that         classification report of the Xception, Inception V3, and the
showed a testing accuracy of above 96% on the rice leaf           MRW-CNN model on the rice leaf dataset is shown in Table 9,
dataset is shown in Fig. 19 and Fig. 20.                          Table 10, and Table 11. The classification report shows the
                                                                  precision, recall, F1-Score, support scores for all the classes
   Fig. 19 shows the normalized confusion matrix of the           in the rice leaf dataset for the test images. It also calculates
model Inception V3 on the rice leaf dataset. It can be observed   the macro average, accuracy, and weighted average.
from the confusion matrix that the class bls performs slightly
better when compared to the other classes. Some classes are       H. WHEAT
confused with each other, like rblast and bls, riceh and ricebs,  The performance of the state-of-the-art deep learning models
etc. Similarly, Fig. 20 shows the normalized confusion matrix     on the wheat leaf dataset is shown in Table 12. Out
for the Xception model on the rice leaf dataset. It can be        of all eight models trained on the wheat leaf dataset,
observed from the figure that all the classes perform equally     the models MobileNetV2, MobileNet, Inception V3, and
the same except for rbb. Some classes that are highly confused    InceptionResNetV2 showed a validation accuracy above
with each other are rblast and bls, riceh, and bls, etc. The      90% when compared to all the other models trained.
proposed MRW-CNN model trained from scratch performed             The MobileNetV2 model showed a validation accuracy of
equally well on the rice leaf dataset with a validation accuracy  0.9724 and a testing accuracy of 0.9632. The MobileNet
of 0.9729 and a testing accuracy of 0.9706 when compared          model showed a validation accuracy of 0.9632 and a testing
with the fine-tuned models. The normalized confusion matrix       accuracy of 0.9628. The Inception V3 model showed a
of the MRW-CNN model on the rice leaf dataset is shown            validation accuracy of 0.9496 and a testing accuracy of
in Fig. 21. In Fig. 21 from the normalized confusion matrix,      0.9508. The InceptionResNetV2 model showed a validation
it can be observed that class ricebs performs slightly better
when compared to the other classes. Confusion among classes                                                                                                      16327

VOLUME 12, 2024
                                                                                       D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease
TABLE 12. Application of the state-of-the-art models on the wheat leaf dataset.

TABLE 13. Classification report of mobilenet model on the wheat leaf
dataset.

TABLE 14. Classification report of MobileNetV2 model on the wheat leaf
dataset.

                                                                        FIGURE 25. Normalized confusion matrix of the model mobilenet on the
                                                                        wheat leaf dataset.

TABLE 15. Classification report of MRW-CNN model on the wheat leaf      testing accuracy of above 96% are shown in Fig. 23 and
dataset.                                                                Fig. 24.

accuracy of 0.9516 and a testing accuracy of 0.9488. The                   The normalized confusion matrix of the fine-tuned models
graphs showing the accuracy and loss performance of the                 that showed a testing accuracy of above 96% on the wheat
fine-tune models trained on the wheat leaf dataset with a               leaf dataset are shown in Fig. 25 and Fig. 26. Fig. 25 shows
                                                                        the normalized confusion matrix of the MobileNet model on
16328                                                                   the wheat leaf dataset. From the confusion matrix, it can
                                                                        be observed that the striperust class performed best when
                                                                        compared to all the other classes in the test data. Confusions
                                                                        among classes also were observed, for example, leafrust
                                                                        and wh, powderymildew and wh, etc. Fig. 26 shows the
                                                                        normalized confusion matrix of the MobileNetV2 model on
                                                                        the wheat leaf dataset.

                                                                           The stripe rust class performed well when compared
                                                                        to others. Some confusing classes were wh and leafrust,
                                                                        powderymildew and wh, leafrust and wh, etc. The proposed
                                                                        MRW-CNN model trained from scratch on the wheat leaf
                                                                        dataset performed well; it showed a validation accuracy of
                                                                        0.9793 and a 2% increase in testing accuracy of 0.9808 when
                                                                        compared to the fine-tuned models. The normalized con-
                                                                        fusion matrix of the MRW-CNN model on the wheat leaf

                                                                                                                                                                         VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

                                                                    FIGURE 28. Accuracy and loss performance of the model MRW-CNN on
                                                                    the wheat leaf dataset.

FIGURE 26. Normalized confusion matrix of the model mobilenetv2 on  support scores for each class in the wheat leaf dataset on the
the wheat leaf dataset.                                             test data. The macro avg, micro average, and accuracy are
                                                                    also calculated.
FIGURE 27. Normalized confusion matrix of the model MRW-CNN on the
wheat leaf dataset.                                                 VII. CASE STUDY OF THE APPLICATION OF THE
                                                                    PROPOSED METHOD
dataset is shown in Fig. 27. In Fig. 27, when observing the         Computer vision technology will be a need in autonomous
diagonal elements, the leafrust, striperust, and wh class of        futuristic farms. CNN based computer vision tasks have
wheat performed equally well when compared to the other             shown great performance in recent years. Therefore, there is
classes. Confusion among classes can also be observed; for          an increased need to concentrate on the practical applications
example, the class wh is highly confused with leafrust, the         of the system developed. If deployed properly, the proposed
leafrust class is confused with wh, etc. The accuracy and           method can help reduce crop loss and increase yield. A plant
loss performance of the MRW-CNN model on the wheat                  protection system APP based on food grains can be developed
leaf dataset is shown in Fig. 28. The classification report of      using the proposed system and the datasets developed. It can
the MobileNet, MobileNetV2, and the MRW-CNN model on                be useful for detecting the diseases affecting the food grains
the wheat leaf dataset is shown in Table 13, Table 14 and           of rice, wheat, and maize.
Table 15. The tables show the precision, recall, F1-score and
                                                                       Depending on the disease detected, appropriate curable
VOLUME 12, 2024                                                     measures can be suggested. Identifying the disease at the
                                                                    correct stage can help the users decide on the amount of
                                                                    pesticide to be used and in what amount. The APP can be
                                                                    trained to identify diseases with similar symptoms by asking
                                                                    users additional questions about the weather conditions, the
                                                                    location of the disease on the leaf, etc. By knowing more
                                                                    about the weather conditions and the location of the disease.
                                                                    It becomes easy to distinguish between diseases showing
                                                                    similar symptoms to some extent.

                                                                       Additional information regarding the weather conditions
                                                                    that can favor the spread or advancement of the disease can
                                                                    be given to the users. Suggestions on the pesticides that can be
                                                                    used and the amount can also be made available. Small-scale
                                                                    growers are the people who benefit more from these kinds
                                                                    of solutions available for diagnosing plant diseases when
                                                                    they are not able to get help from experts all the time. The
                                                                    overall architecture of the plant protection system that can be
                                                                    developed is shown in Fig. 29.

                                                                       Fig. 29 discusses the possible use cases the user can
                                                                    give the plant disease protection system. In case 1, the user
                                                                    provides a diseased leaf to the system to detect the disease
                                                                    affecting the leaf along with its stage. After the detection

                                                                                                                                                                   16329
                                                D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

FIGURE 29. Application of the proposed system.

of the disease and its stage. The system also provides             VIII. CONCLUSION AND FUTURE WORK
suggestions regarding the curable measures and alerts on the       A. CONCLUSION
weather conditions that can be favorable for advancing the         In the current work, real-time datasets for the food grains,
disease. In case 2, when a diseased leaf has similar symptoms      rice, wheat, and maize were formed by considering images
of one or two diseases, more questions can be asked to the         from different datasets and internet sources. Importance
users about the weather and geographical conditions. Based         was given to real-life images of different disease severity
on this information, possible predictions of diseases can be       levels, which is important for farmers or growers to make
made by the system. Also, suggestions for curable measures         appropriate decisions at the early stages of infection.
for these predicted outcomes can be made.                          Identification of disease at the early stage of infection can
                                                                   reduce yield loss and financial loss.
   In case 3, when the system encounters an unknown disease,
options can be given to the user to send it to an expert or plant     According to experimental results, CNN-based deep
pathologist for analysis. The system learns from the disease       learning models pre-trained on the ImageNet dataset, with
identified by the experts and their suggestions. The system        deep transfer learning strategy, downloaded from Keras and
learns from the communication between the users or growers         fine-tuned, and the proposed MRW-CNN model performs
and the experts.                                                   well in identifying and classifying rice, maize, and wheat

16330                                                                                                                                                               VOLUME 12, 2024
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

diseases considered in the current work. All the models were                        [9] H. Hussain and P. S. Tamizharasan, ``The impact of cascaded optimizations
trained for 100 epochs.                                                                   in CNN models and end-device deployment,'' in Proc. 20th ACM Conf.
                                                                                          Embedded Networked Sensor Syst., J. Gummeson, S. I. Lee, J. Gao, and
   During the development of the dataset, real-life images                                G. Xing, Eds. Boston, MA, USA, 2022, pp. 954­961.
were concentrated on more, rather than processed images.
Selected images from the datasets were pre-processed to                            [10] G. Zhou, W. Zhang, A. Chen, M. He, and X. Ma, ``Rapid detection of rice
remove foreign objects if present. The number of images                                   disease based on FCM-KM and faster R-CNN fusion,'' IEEE Access, vol. 7,
collected under each class was 100 which was not enough                                   pp. 143190­143206, 2019.
for training a deep learning model. So, the technique of data
augmentation was applied to all images per class to increase                       [11] Z. Libo, H. Tian, G. Chunyun, and M. Elhoseny, ``Real-time detection of
the size of the dataset. The original images were combined                                Cole diseases and insect pests in wireless sensor networks,'' J. Intell. Fuzzy
with the augmented images, and then the dataset formed                                    Syst., vol. 37, no. 3, pp. 3513­3524, Oct. 2019.
was splitted 80% for training and 10% each for validation
and testing for all three food grains: rice, wheat, and maize.                     [12] P. Jiang, Y. Chen, B. Liu, D. He, and C. Liang, ``Real-time detection
The datasets were then applied to different fine-tuned CNN                                of apple leaf diseases using deep learning approach based on improved
models and the proposed MRW-CNN model. The models                                         convolutional neural networks,'' IEEE Access, vol. 7, pp. 59069­59080,
were evaluated using performance metrics like accuracy,                                   2019.
f1-score, precision, recall, etc.
                                                                                   [13] M. Shoaib, B. Shah, S. Ei-Sappagh, A. Ali, A. Ullah, F. Alenezi, T. Gechev,
B. FUTURE WORK                                                                            T. Hussain, and F. Ali, ``An advanced deep learning models-based plant
It can be observed that the datasets we developed have some                               disease detection: A review of recent research,'' Frontiers Plant Sci.,
limitations. Although most of the images in the dataset are                               vol. 14, Mar. 2023, Art. no. 1158933.
real-life, some are pre-processed. In real-life scenarios, more
factors will influence the results than those concentrated                         [14] M. Shoaib, B. Shah, T. Hussain, A. Ali, A. Ullah, F. Alenezi, T. Gechev,
on developing the datasets in the current work. Therefore,                                F. Ali, and I. Syed, ``A deep learning-based model for plant lesion
as future work, more importance can be given to factors like                              segmentation, subtype identification, and survival probability estimation,''
how the impact of climate change can affect plant diseases                                Frontiers Plant Sci., vol. 13, Dec. 2022, Art. no. 1095547.
collection of leaf diseased images during different climatic
conditions.                                                                        [15] M. Akbar, M. Ullah, B. Shah, R. U. Khan, T. Hussain, F. Ali, F. Alenezi,
                                                                                          I. Syed, and K. S. Kwak, ``An effective deep learning approach for the
   Some of the other future works can be annotated images                                 classification of bacteriosis in peach leave,'' Frontiers Plant Sci., vol. 13,
from the dataset can be used in object detection models like                              p. 4723, Nov. 2022.
MaskRCNN to detect disease severity levels. A CNN based
deep learning system can be developed to detect and classify                       [16] D. P. Hughes and M. Salathé, ``An open access repository of images
diseases in food grains; the dataset can be further extended                              on plant health to enable the development of mobile disease diagnostics
by adding images of more food grains, images of multiple                                  through machine learning and crowdsourcing,'' 2015, arXiv:1511.08060.
diseases on the same leaf, etc.
                                                                                   [17] Q. Zeng, X. Ma, B. Cheng, E. Zhou, and W. Pang, ``GANs-based data
REFERENCES                                                                                augmentation for citrus disease severity detection using deep learning,''
                                                                                          IEEE Access, vol. 8, pp. 172882­172891, 2020.
 [1] J. G. A. Barbedo, ``A review on the main challenges in automatic plant
       disease identification based on visible range images,'' Biosystems Eng.,    [18] J. Ma, K. Du, F. Zheng, L. Zhang, Z. Gong, and Z. Sun, ``A recognition
       vol. 144, pp. 52­60, Apr. 2016.                                                    method for cucumber diseases using leaf symptom images based on deep
                                                                                          convolutional neural network,'' Comput. Electron. Agricult., vol. 154,
 [2] D. S. Joseph, P. M. Pawar, and R. Pramanik, ``Intelligent plant disease              pp. 18­24, Nov. 2018.
       diagnosis using convolutional neural network: A review,'' Multimedia
       Tools Appl., vol. 82, no. 14, pp. 21415­21481, Jun. 2023.                   [19] Q. Wu, Y. Chen, and J. Meng, ``DCGAN-based data augmenta-
                                                                                          tion for tomato leaf disease identification,'' IEEE Access, vol. 8,
 [3] J. Chen, J. Chen, D. Zhang, Y. Sun, and Y. A. Nanehkaran, ``Using deep               pp. 98716­98728, 2020.
       transfer learning for image-based plant disease identification,'' Comput.
       Electron. Agricult., vol. 173, Jun. 2020, Art. no. 105393.                  [20] E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, ``A comparative study of
                                                                                          fine-tuning deep learning models for plant disease identification,'' Comput.
 [4] G. Wang, Y. Sun, and J. Wang, ``Automatic image-based plant disease                  Electron. Agricult., vol. 161, pp. 272­279, Jun. 2019.
       severity estimation using deep learning,'' Comput. Intell. Neurosci.,
       vol. 2017, pp. 1­8, Jul. 2017.                                              [21] R. Kumar, U. Mina, R. Gogoi, A. Bhatia, and R. C. Harit, ``Effect of
                                                                                          elevated temperature and carbon dioxide levels on maydis leaf blight
 [5] O. Getch. Rice Knowledge Bank: Your Information Source for Rice                      disease tolerance attributes in maize,'' Agricult., Ecosystems Environ.,
       Farming. Accessed: Jan. 30, 2023. [Online]. Available: http://www.                 vol. 231, pp. 98­104, Sep. 2016.
       knowledgebank.irri.org/step-by-step-production/growth/pests-and-
       diseases/diseases                                                           [22] S. Nigam, R. Jain, S. Marwaha, A. Arora, M. A. Haque, A. Dheeraj, and
                                                                                          V. K. Singh, ``Deep transfer learning model for disease identification in
 [6] E. D. Wolf and J. P. Shroyer. Wheat Disease Identification. Accessed:                wheat crop,'' Ecological Informat., vol. 75, Jul. 2023, Art. no. 102068.
       Nov. 25, 2022. [Online]. Available: https://www.bookstore.ksre.ksu.edu/
       pubs/MF2994.pdf                                                             [23] M. K. Vishal, D. Tamboli, A. Patil, R. Saluja, B. Banerjee, A. Sethi,
                                                                                          D. Raju, S. Kumar, R. N. Sahoo, V. Chinnusamy, and J. Adinarayana,
 [7] S. Mahadik, P. M. Pawar, and R. Muthalagu, ``Efficient intelligent intrusion         ``Image-based phenotyping of diverse rice (Oryza sativa L.) genotypes,''
       detection system for heterogeneous Internet of Things (HetIoT),'' J. Netw.         2020, arXiv:2004.02498.
       Syst. Manag., vol. 31, no. 1, p. 2, Jan. 2023.
                                                                                   [24] A. Chug, A. Bhatia, A. P. Singh, and D. Singh, ``A novel frame-
 [8] K. Syama, J. A. A. Jothi, and N. Khanna, ``Automatic disease prediction              work for image-based plant disease detection using hybrid deep
       from human gut metagenomic data using boosting GraphSAGE,'' BMC                    learning approach,'' Soft Comput., vol. 27, no. 18, pp. 13613­13638,
       Bioinf., vol. 24, no. 1, p. 126, Mar. 2023.                                        Sep. 2023.

VOLUME 12, 2024                                                                    [25] M. A. Haque, S. Marwaha, C. K. Deb, S. Nigam, A. Arora,
                                                                                          K. S. Hooda, P. L. Soujanya, S. K. Aggarwal, B. Lall, M. Kumar, S. Islam,
                                                                                          M. Panwar, P. Kumar, and R. C. Agrawal, ``Deep learning-based approach
                                                                                          for identification of diseases of maize crop,'' Sci. Rep., vol. 12, no. 1,
                                                                                          p. 6334, Apr. 2022.

                                                                                   [26] S. Das, A. Biswas, and P. Sinha, ``Deep learning analysis of rice blast
                                                                                          disease using remote sensing images,'' IEEE Geosci. Remote Sens. Lett.,
                                                                                          vol. 20, pp. 1­5, 2023.

                                                                                   [27] M. Shoaib, T. Hussain, B. Shah, I. Ullah, S. M. Shah, F. Ali, and S. H. Park,
                                                                                          ``Deep learning-based segmentation and classification of leaf images for
                                                                                          detection of tomato plant disease,'' Frontiers Plant Sci., vol. 13, Oct. 2022,
                                                                                          Art. no. 1031748.

                                                                                   [28] X. Zhang, Y. Qiao, F. Meng, C. Fan, and M. Zhang, ``Identification of
                                                                                          maize leaf diseases using improved deep convolutional neural networks,''
                                                                                          IEEE Access, vol. 6, pp. 30370­30377, 2018.

                                                                                                                                                                                  16331
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

[29] H. Wu, T. Wiesner-Hanks, E. L. Stewart, C. DeChant, N. Kaczmar,                 [55] Common Corn Rust Puccinia Sorghi Schwein. Accessed: Oct. 7, 2022.
       M. A. Gore, R. J. Nelson, and H. Lipson, ``Autonomous detection of plant             [Online]. Available: https://www.forestryimages.org/browse/detail.cfm?
       disease symptoms directly from aerial imagery,'' Plant Phenome J., vol. 2,           imgnum=1538028
       no. 1, pp. 1­9, Jan. 2019.
                                                                                     [56] A. Sisson. Common Corn Rust Puccinia Sorghi Schwein. Accessed:
[30] J. Sun, Y. Yang, X. He, and X. Wu, ``Northern maize leaf blight detection              Oct. 7, 2022. [Online]. Available: https://www.forestryimages.
       under complex field environment based on deep learning,'' IEEE Access,               org/browse/detail.cfm?imgnum=5608271
       vol. 8, pp. 33679­33688, 2020.
                                                                                     [57] G. Holmes. Common Corn Rust Puccinia Sorghi Schwein.
[31] S. Mishra, R. Sachan, and D. Rajpal, ``Deep convolutional neural network               Accessed: Oct. 7, 2022. [Online]. Available: https://www.
       based detection system for real-time corn plant disease recognition,'' Proc.         forestryimages.org/browse/detail.cfm?imgnum=5606719
       Comput. Sci., vol. 167, pp. 2003­2010, Jan. 2020.
                                                                                     [58] A. Robertson. Gray Leaf Spot. Accessed: Oct. 7, 2022. [Online].
[32] H. T. Rauf, B. A. Saleem, M. I. U. Lali, M. A. Khan, M. Sharif, and                    Available: https://crops.extension.iastate.edu/cropnews/2018/06/weather-
       S. A. C. Bukhari, ``A citrus fruits and leaves dataset for detection and             conditions-brown-spot-and-node-rot-and-gray-leaf-spot
       classification of citrus diseases through machine learning,'' Data Brief,
       vol. 26, Oct. 2019, Art. no. 104340.                                          [59] Northern Corn Leaf Blight, Grey Leaf Spot Top Ontario
                                                                                            Corn Diseases. Accessed: Nov. 22, 2022. [Online]. Available:
[33] J. Lu, J. Hu, G. Zhao, F. Mei, and C. Zhang, ``An in-field automatic                   https://www.topcropmanager.com/northerncornleafblightgreyleafspottop-
       wheat disease diagnosis system,'' Comput. Electron. Agricult., vol. 142,             ontariocorndiseases19935/
       pp. 369­379, Nov. 2017.
                                                                                     [60] Managing Northern Corn Leaf Blight. Accessed: Oct. 7, 2022. [Online].
[34] R. Qiu, C. Yang, A. Moghimi, M. Zhang, B. J. Steffenson, and C. D. Hirsch,             Available: https://www.cropscience.bayer.ca/articles/2021/northern-corn-
       ``Detection of fusarium head blight in wheat using a deep neural network             leaf-blight
       and color imaging,'' Remote Sens., vol. 11, no. 22, p. 2658, Nov. 2019.
                                                                                     [61] Chroanch. Northern Corn Leaf Blight. Accessed: Oct. 7, 2022.
[35] Y. Lu, S. Yi, N. Zeng, Y. Liu, and Y. Zhang, ``Identification of rice diseases         [Online]. Available: https://commons.wikimedia.org/Northern_
       using deep convolutional neural networks,'' Neurocomputing, vol. 267,                corn_leaf_blight.JPG
       pp. 378­384, Dec. 2017.
                                                                                     [62] D. Mueller. Northern Corn Leaf Blight of Corn. Accessed: Oct. 7, 2022.
[36] F. Jiang, Y. Lu, Y. Chen, D. Cai, and G. Li, ``Image recognition of four rice          [Online]. Available: https://cropprotectionnetwork.org/encyclopedia/
       leaf diseases based on deep learning and support vector machine,'' Comput.           northern-corn-leaf-blight-of-corn
       Electron. Agricult., vol. 179, Dec. 2020, Art. no. 105824.
                                                                                     [63] A. Robertson. Northern Corn Leaf Blight of Corn. Accessed:
[37] D. P. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra,                     Oct. 7, 2022. [Online]. Available: https://cropprotectionnetwork.
       ``PlantDoc: A dataset for visual plant disease detection,'' in Proc. 7th ACM         org/encyclopedia/northern-corn-leaf-blight-of-corn
       IKDD CoDS 25th COMAD, 2019, pp. 249­253.
                                                                                     [64] Southern Rust. Accessed: Oct. 7, 2022. [Online]. Available:
[38] S. Riyaz. Rice Leafs: An Image Collection Four Rice Diseases.                          https://cropwatch.unl.edu/plantdisease/corn/southern-rust
       Accessed: Jan. 30, 2023. [Online]. Available: https://www.kaggle.
       com/datasets/shayanriyaz/riceleafs                                            [65] Southern Rust of Corn. Accessed: Oct. 7, 2022. [Online]. Available:
                                                                                            https://www.pioneer.com/us/agronomy/southern_rust_cropfocus.html
[39] O. Getch. Wheat Leaf Dataset: Disease Affected and Healthy Wheat
       Leaf. Accessed: Jan. 30, 2023. [Online]. Available: https://www.              [66] Mississippi Crop Situation. Accessed: Oct. 7, 2022. [Online]. Avail-
       kaggle.com/datasets/olyadgetch/wheat-leaf-dataset                                    able: https://www.mississippi-crops.com/2012/07/21/southern-corn-rust-
                                                                                            is-a-late-fungicide-application-warranted/
[40] Rice Knowledge Bank. Accessed: Oct. 12, 2022. [Online]. Available: http://
       www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-       [67] C. A. Schneider, W. S. Rasband, and K. W. Eliceiri, ``NIH image to ImageJ:
       sheets                                                                               25 years of image analysis,'' Nature Methods, vol. 9, no. 7, pp. 671­675,
                                                                                            Jul. 2012.
[41] D. Groth. Brown Spot of Rice. Accessed: Oct. 6, 2022. [Online]. Available:
       https://www.forestryimages.org/browse/detail.cfm?imgnum=5410673               [68] Imagej. Accessed: Oct. 6, 2022. [Online]. Available:
                                                                                            https://imagej.net/software/imagej/
[42] Leaf Rust. Accessed: Oct. 12, 2022. [Online]. Available: https://
       agritech.tnau.ac.in/crop_protection/wheat/crop_prot_crop                      [69] Make Sense Ai Tool. Accessed: Oct. 6, 2022. [Online]. Available:
                                                                                            https://www.makesense.ai/
[43] Leaf Rust. Accessed: Oct. 12, 2022. [Online]. Available:
       https://www.cropscience.bayer.co.nz/pests/diseases/leaf-rust--wheat           [70] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang,
                                                                                            A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei,
[44] Leaf Rust of Wheat. Accessed: Oct. 12, 2022. [Online]. Available:                      ``ImageNet large scale visual recognition challenge,'' Int. J. Comput. Vis.,
       https://extensionaus.com.au/FieldCropDiseasesVic/docs/identification-                vol. 115, no. 3, pp. 211­252, Dec. 2015.
       management-of-field-crop-diseases-in-victoria/foliar-diseases-of-
       wheat/leaf-rust-of-wheat/                                                     [71] I. Chelliah. Confusion Matrix for Multiclass Classification. Accessed:
                                                                                            Feb. 21, 2023. [Online]. Available: https://medium.com/mlearning-
[45] Powdery Mildew. Accessed: Oct. 12, 2022. [Online]. Available:                          ai/confusion-matrix-for-multiclass-classification-f25ed7173e66
       https://www.cropscience.bayer.co.nz/pests/diseases/powdery-mildew--
       wheat                                                                         [72] Micro-Average & Macro-Average Scoring Metrics--Python. Accessed:
                                                                                            Feb. 21, 2023. [Online]. Available: https://vitalflux.com/micro-average-
[46] New Permits Aid Control of Powdery Mildew in Wheat. Accessed: Oct. 12,                 macro-average-scoring-metrics-multi-class-classification/
       2022. [Online]. Available: https://www.graincentral.com/cropping/new-
       permits-aid-control-of-powdery-mildew-in-wheat/                               [73] F1 Score in Machine Learning: Intro & Calculation. Accessed: Feb. 21,
                                                                                            2023. [Online]. Available: https://www.v7labs.com/blog/f1-score-guide
[47] Scout Wheat Fields for Early Disease Detection. Accessed: Oct. 12,
       2022. [Online]. Available: https://cropwatch.unl.edu/2021/scout-wheat-        [74] R. R. Selvaraju, P. Chattopadhyay, M. Elhoseiny, T. Sharma, D. Batra,
       fields-early-disease-detection                                                       D. Parikh, and S. Lee, ``Choose your neuron: Incorporating domain
                                                                                            knowledge through neuron-importance,'' in Proc. Eur. Conf. Comput. Vis.,
[48] Powdery Mildew Symptoms. Accessed: Oct. 12, 2022. [Online]. Available:                 2018, pp. 526­541.
       https://doraagri.com/powdery-mildew-on-plants/
                                                                                                                    DIANA SUSAN JOSEPH (Student Member,
[49] Stripe Rust (Yellow Rust) of Wheat. Accessed: Oct. 12, 2022.                                                   IEEE) received the B.Tech. degree in computer
       [Online]. Available: https://extension.uga.edu/publications/detail.html?                                     science and engineering from the Rajagiri School
       number=C960&                                                                                                 of Engineering and Technology, in 2008, and the
                                                                                                                    M.Tech. degree in computer and communication
[50] Tan Spot. Accessed: Oct. 12, 2022. [Online]. Available: https://                                               from Karunya University, Coimbatore, in 2011.
       andersonscanada.com/2017/05/17/wheat-leaf-diseases-2/                                                        She is currently pursuing the Ph.D. degree in
                                                                                                                    computer science and engineering with the Birla
[51] Tan Spot. Accessed: Oct. 12, 2022. [Online]. Available:                                                        Institute of Technology and Science, Dubai. She
       https://extension.okstate.edu/programs/digital-diagnostics/plant-                                            was an Assistant Professor with the Department of
       diseases/tan-spot.html                                                        Computer Science and Engineering, in 2012. Her research interests include
                                                                                     image processing, machine learning, and deep learning.
[52] Tan Spot of Wheat. Accessed: Oct. 12, 2022. [Online]. Available:
       https://extensionpublications.unl.edu/assets/html/g429/build/g429.htm                                                                                                          VOLUME 12, 2024

[53] Tan Spot. Accessed: Oct. 12, 2022. [Online]. Available:
       https://cropscience.bayer.co.uk/threats/diseases/cereal-diseases/tan-spot/

[54] Common Rust of Corn. Accessed: Oct. 7, 2022. [Online]. Available:
       https://ohioline.osu.edu/factsheet/plpath-cer-02

16332
D. S. Joseph et al.: Real-Time Plant Disease Dataset Development and Detection of Plant Disease

                               PRANAV M. PAWAR (Member, IEEE) received              at international level. He has published more than 40 papers at national and
                               the Graduate degree in computer engineering          international level. He is also IBM DB2 and IBM RAD certified professional
                               from Dr. Babasaheb Ambedkar Technological            and completed NPTEL certification in different subjects. His research
                               University, Maharashtra, India, in 2005, the mas-    interests include energy efficient MAC for WSN, QoS in WSN, wireless
                               ter's degree in computer engineering from Pune       security, green technology, computer architecture, database management
                               University, in 2007, and the Ph.D. degree in         systems, and bioinformatics.
                               wireless communication from Aalborg University,
                               Denmark, in 2016. His Ph.D. thesis received                                         KAUSTUBH CHAKRADEO received the Bach-
                               nomination for Best Thesis Award from Aalborg                                       elor of Engineering degree in information tech-
                               University. Currently, he is an Assistant Professor                                 nology from Savitribai Phule Pune University
with the Department of Computer Science, Birla Institute of Technology                                             (formerly Pune University), in 2019, and the Mas-
and Science, Dubai, before to BITS, he was a Postdoctoral Fellow with                                              ter of Science degree in artificial intelligence from
Bar-Ilan University, Israel, from March 2019 to October 2020, in an                                                Radboud University, The Netherlands, in 2022.
area of wireless communication and deep leaning. He was recipient of                                               He is currently pursuing the Ph.D. degree with the
Outstanding Postdoctoral Fellowship from Israel Planning and Budgeting                                             Section of Epidemiology, Department of Public
Committee. He was an Associate Professor with MIT ADT University, Pune,                                            Health, University of Copenhagen. His research
from 2018 to 2019, and also as an Associate Professor with the Department                                          interests include computer vision, deep learning,
of Information Technology, STES's Shrimati Kashibai Navale College of               AI ethics, and infectious disease modeling.
Engineering, Pune, from 2008 to 2018. From 2006 to 2007, he was a
System Executive with POS-IPC, Pune, India. He received Recognition from
Infosys Technologies Ltd., for contribution in Campus Connect Program
and also received different funding for research and attending conferences

VOLUME 12, 2024                                                                                  16333

---

## EXTRACTED FIGURES (visual analysis)

### Figure 9 â€” "Architecture of proposed CNN model MRW-CNN." (page 13, `papers/04- Real-Time Plant Disease Dataset Development/inputs/figures/page13.png`, high-res crop: `figures/fig9_crop.png`, `figures/fig9_left.png`)

Diagram traced via rendered PNG (zoom 6-10x), arrows followed precisely to confirm block order:

```
Input Training Images
  -> Convolution(32, 3x3) -> BatchNormalization -> ReLU      [block 1]
  -> Convolution(64, 3x3) -> BatchNormalization -> ReLU      [block 2]
  -> Convolution(64, 3x3) -> BatchNormalization -> ReLU      [block 3]
  -> Convolution(256, 3x3) -> BatchNormalization -> ReLU     [block 4]
  -> Convolution(256, 3x3) -> BatchNormalization -> ReLU     [block 5]
  -> Global Average Pooling
  -> Dense layer
  -> Softmax
  -> Output Classes
```

Confirmed details visible in the figure:
- Exactly 5 conv blocks, each labeled `Convolution (<filters>,3x3)` followed by `Batch Normalization` then `ReLU`.
- Filters in order: **32, 64, 64, 256, 256** (NOT a smooth doubling progression â€” block 2 and 3 both use 64, block 4 and 5 both use 256).
- Kernel size for every block: **3x3** (explicitly labeled in each box).
- Arrow path: block1 -> block2 -> block3 (straight down on the left column), then block3 -> block4 (routed right then up to the right column's lower box), then block4 -> block5 (straight up), then block5 -> Global Average Pooling (right).
- After Global Average Pooling: a single Dense layer, then Softmax, then "Output Classes".

NOT visible in the figure (confirms these are genuinely absent from the paper, not just missed in text extraction):
- No MaxPooling / pooling layer of any kind is drawn between or after conv blocks (only GAP at the very end).
- No stride value is annotated on any block (only `(filters,3x3)` kernel notation).
- No padding mode is annotated.

This matches the body text in Section V ("The model consists of five Convolution layers, each followed by BatchNormalization and Rectified Linear Unit (ReLU)... Global Average Pooling layer is added... followed by a Dense layer to map the extracted features... Softmax activation function").
