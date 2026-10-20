### 4.6. Preprocessing and Inference Latency Evaluation in a Laptop-Camera Application

ChiliEdgeNet was integrated into a laptop camera application, where the user places a leaf inside a region of interest (ROI) and triggers a prediction on each frame. This experiment evaluates only preprocessing and inference latency as a proof of integration feasibility; it does not use camera frames to estimate real-field accuracy.

The test session comprises 62 predictions, run at batch size 1 on a 13th Gen Intel Core i7-13650HX CPU. The measured interval starts from converting the ROI's RGB image to PIL, continuing through resize, normalization, model inference, softmax, and top-1 prediction. The measurement excludes time for reading frames from the camera, cropping the ROI, converting BGR to RGB, rendering the UI, or writing files.

**Table 11.** Preprocessing and inference latency of ChiliEdgeNet on a laptop CPU.

| Metric | Latency (ms) |
|---|---:|
| Number of measurements | 62 |
| Mean | 32.18 |
| Median | 29.70 |
| 95th percentile | 46.49 |
| Minimum | 17.33 |
| Maximum | 122.76 |

A median of 29.70 ms and a 95th percentile of 46.49 ms show that the preprocessing-and-inference stage can respond quickly on a laptop CPU without a GPU. However, the results come from a single measurement session on a single device and do not include the application's full capture-to-display cost; the test device is also not dedicated edge hardware. This experiment therefore provides only preliminary evidence of ChiliEdgeNet's feasibility for integration into a camera application, not a representative measurement of the full pipeline or of dedicated edge hardware. The labelled data in the demo session is unbalanced and contains many repeated frames of the same subject, so accuracy and a confusion matrix are not reported in this section.
