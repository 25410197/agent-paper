## 6. Conclusion

At seed 42, ChiliEdgeNet reaches **98.93% accuracy**, macro-F1 **0.9897**, using **671,173 parameters**, with CPU inference at **20.24 ms/image**. Compared with RTR_Lite_MobileNet (Duhan et al., 2025) at the same seed, the model is **0.80 pp (3 images)** higher and uses about **35% fewer parameters**; since each architecture was run only once and the confidence intervals overlap, the accuracy advantage should be read as a trend rather than an established statistical ranking.

The ablation results show:

1. Disabling MobileNetV2's residual connections reduces accuracy by 2.41 pp on clean images; the difference remains significant after Holm correction for the two direct cuts.
2. The Tiny Transformer shows no advantage on clean images or mild degradation, but the gap grows under severe degradation. At brightness factor 0.2, the full model is 3.75 pp higher, and the difference is significant after Holm correction.
3. Triplet Attention and `RES_Block` do not provide a stable benefit. The current model, which omits both, matches or exceeds the add-back configurations, but the contrasts are not significant after Holm correction.
4. 2D-RoPE, GQA, and RMSNorm together avoid 41,856 parameters relative to using all three standard alternatives simultaneously; single-seed results do not show an accuracy advantage on clean images.
5. On the clean 1,463-image cross-dataset query set, adaptation with 42 support images raises accuracy from 30.96% to 89.75% (+58.78 pp); a paired McNemar test records 870 improvements and 10 regressions ($p=1.83\times10^{-242}$). This result applies only to the four mapped classes.

Future directions include repeating the experiments across multiple seeds, checking for duplicate/near-duplicate images across splits, extending the stress test to real acquisition degradations, and measuring latency and memory on physical edge hardware with a real quantization backend.
