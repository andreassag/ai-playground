# PyTorch Deep Learning Pipelines: Vision, Audio & NLP

A curated collection of end-to-end, production-grade PyTorch deep learning pipelines spanning computer vision, speech/audio processing, and natural language processing. Each pipeline is implemented as a self-contained, mathematically rigorous Jupyter notebook featuring custom neural architectures, robust data pipelines, modern training techniques, clean nested `tqdm` progress tracking, best-model checkpointing, and inline TensorBoard telemetry.

---

## Notebook Portfolio

### Computer Vision (Torchvision Datasets)

| Notebook | Task | Architecture | Dataset | Key Techniques |
| :--- | :--- | :--- | :--- | :--- |
| [`00_mnist_classification_resnet.ipynb`](notebooks/vision/00_mnist_classification_resnet.ipynb) | Handwritten Digit Recognition | ResNet + Squeeze-and-Excitation (SE) | MNIST | Moment deskewing, affine/perspective transforms, OneCycleLR, Label Smoothing, Test-Time Augmentation (TTA), TPU/XLA support |
| [`01_image_classification_eurosat_convnext.ipynb`](notebooks/vision/01_image_classification_eurosat_convnext.ipynb) | Satellite Land Cover Classification | ConvNeXt + Squeeze-and-Excitation (SE) + LayerNorm | EuroSAT (10 classes) | 7x7 Depthwise separable convolutions, Inverted bottlenecks, Label Smoothing, OneCycleLR, Grad-CAM attribution heatmaps, TPU/XLA support |
| [`02_object_detection_voc_retinanet.ipynb`](notebooks/vision/02_object_detection_voc_retinanet.ipynb) | Multi-Class Object Detection | Feature Pyramid Network (FPN) + Anchor-Free Head | Pascal VOC 2007 (20 classes) | Multi-scale FPN (P3, P4, P5), Focal Loss, Generalized IoU (GIoU) loss, Centerness quality branch, Non-Maximum Suppression (NMS), TPU/XLA support |
| [`03_semantic_segmentation_voc_deeplabv3.ipynb`](notebooks/vision/03_semantic_segmentation_voc_deeplabv3.ipynb) | Semantic Scene Segmentation | DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) | Pascal VOC 2007 (21 classes) | Multi-rate atrous convolutions ($r \in \{6, 12, 18\}$), Low-level decoder fusion, Composite Cross-Entropy + Soft Dice Loss, Mean IoU (mIoU) metrics, TPU/XLA support |
| [`04_optical_flow_flyingchairs_flownet.ipynb`](notebooks/vision/04_optical_flow_flyingchairs_flownet.ipynb) | Dense 2D Optical Flow Estimation | FlowNetCorr (Siamese Encoders + Patch Correlation Layer) | FlyingChairs / KittiFlow | 2D Cross-frame correlation layer, Multi-scale flow refinement decoder, Multi-scale End-Point Error (EPE) loss, Middlebury color-wheel flow maps, TPU/XLA support |
| [`05_stereo_matching_sceneflow_stereonet.ipynb`](notebooks/vision/05_stereo_matching_sceneflow_stereonet.ipynb) | Binocular Stereo Disparity & Depth Estimation | 3D Cost Volume Matching CNN + Soft-Argmin Regression | SceneFlow / CREStereo | 3D Disparity correlation cost volume, 3D Hourglass filtering, Differentiable Soft-Argmin operator, Smooth L1 disparity loss, Red-Cyan 3D Anaglyphs, TPU/XLA support |
| [`06_image_pairs_lfw_siamese.ipynb`](notebooks/vision/06_image_pairs_lfw_siamese.ipynb) | Face Verification & Metric Learning | Dual-Stream Siamese ResNet + $L_2$-Normalized Hypersphere | LFWPairs (6,000 pairs) | Contrastive Margin Loss, Cosine Embedding Loss, ROC curves, Area Under Curve (AUC-ROC), Equal Error Rate (EER) calibration, TPU/XLA support |
| [`07_image_captioning_coco_transformer.ipynb`](notebooks/vision/07_image_captioning_coco_transformer.ipynb) | Vision-Language Image Captioning | CNN/ViT Feature Grid Encoder + Autoregressive Transformer Decoder | CocoCaptions / Flickr8k | Causal self-attention, Visual token cross-attention, Dynamic padding collator, Greedy & Beam Search decoding, Corpus BLEU-1/4, TPU/XLA support |
| [`08_video_classification_ucf101_r2plus1d.ipynb`](notebooks/vision/08_video_classification_ucf101_r2plus1d.ipynb) | Spatiotemporal Action Recognition | Factorized Spatiotemporal (2+1)D ResNet (R(2+1)D) | UCF101 / HMDB51 | Factorized 2D spatial + 1D temporal convolutions, 16-frame uniform temporal sampling, Top-1/Top-5 Video Accuracy, Spatiotemporal action matrix, TPU/XLA support |
| [`09_video_prediction_movingmnist_convlstm.ipynb`](notebooks/vision/09_video_prediction_movingmnist_convlstm.ipynb) | Spatiotemporal Video Forecasting & Frame Synthesis | Encoder-Decoder Spatiotemporal ConvLSTM | MovingMNIST (20 frames) | Convolutional LSTM recurrent cells, Multi-scale Composite Loss (MSE + SSIM), Rollout PSNR/SSIM diagnostics, Side-by-side animated frame rollout, TPU/XLA support |

### Speech & Audio Processing

| Notebook | Task | Architecture | Dataset | Key Techniques |
| :--- | :--- | :--- | :--- | :--- |
| [`01_speech_commands_convnext.ipynb`](notebooks/audio/01_speech_commands_convnext.ipynb) | Keyword Spotting (KWS) | 1D ConvNeXt + SE Attention + Multi-Head Temporal Attention | SpeechCommands v2 (35 classes) | Log-Mel Filterbanks, SpecAugment, OneCycleLR, Cosine Annealing, Interactive Audio Inference, Cloud TPU / XLA support |
| [`02_librispeech_conformer_ctc.ipynb`](notebooks/audio/02_librispeech_conformer_ctc.ipynb) | End-to-End Speech Recognition (ASR) | Conformer (Convolution-Augmented Transformer) + CTC | LibriSpeech (`dev-clean`) | Character Tokenizer, Dynamic Collate Padding, CTC Loss, Levenshtein CER/WER evaluation, Cloud TPU / XLA support |
| [`03_speech_denoising_conv_tasnet.ipynb`](notebooks/audio/03_speech_denoising_conv_tasnet.ipynb) | Time-Domain Speech Enhancement & Denoising | Conv-TasNet (Time-Domain Audio Separation Network) | LibriSpeech + Synthetic Noise | Dilated 1D Depthwise Convolutions, SI-SDR Loss, SDR Gain metrics, Interactive A/B Audio Player, Cloud TPU / XLA support |
| [`04_speaker_verification_ecapa_tdnn.ipynb`](notebooks/audio/04_speaker_verification_ecapa_tdnn.ipynb) | Voice Biometrics & Speaker Verification | ECAPA-TDNN + Attentive Statistics Pooling | LibriSpeech (Multi-Speaker) | Additive Angular Margin (ArcFace) Loss ($s=30, m=0.2$), Equal Error Rate (EER), DET/ROC Curves, 2D t-SNE, Cloud TPU / XLA support |

### Natural Language Processing (TorchText Datasets)

| Notebook | Task | Architecture | Dataset | Key Techniques |
| :--- | :--- | :--- | :--- | :--- |
| [`01_news_classification_textcnn_attention.ipynb`](notebooks/text/01_news_classification_textcnn_attention.ipynb) | Multi-Class News Classification | Multi-Scale TextCNN + Squeeze-and-Excitation + Multi-Head Attention Pooling | AG_NEWS (4 classes) | Multi-scale 1D Convolutions ($k \in \{2, 3, 4, 5\}$), Label Smoothing, OneCycleLR, Saliency Attention Attribution, TPU/XLA support |
| [`02_causal_language_model_transformer.ipynb`](notebooks/text/02_causal_language_model_transformer.ipynb) | Autoregressive Causal Language Modeling | Decoder Transformer + Rotary Position Embeddings (RoPE) + SwiGLU + Pre-RMSNorm | WikiText-2 | Causal Masking, Tied Embeddings, Perplexity ($\text{PPL}$) metric, Temperature/Top-$p$/Top-$k$ Sampling, TPU/XLA support |
| [`03_machine_translation_seq2seq_transformer.ipynb`](notebooks/text/03_machine_translation_seq2seq_transformer.ipynb) | Neural Machine Translation (English $\to$ German) | Full Seq2Seq Transformer (Encoder-Decoder + Cross-Attention) | Multi30k | Sinusoidal Positional Encoding, Dynamic Padding Masks, Greedy & Beam Search Decoding, Corpus BLEU-4, TPU/XLA support |
| [`04_sequence_tagging_bilstm_crf.ipynb`](notebooks/text/04_sequence_tagging_bilstm_crf.ipynb) | POS Tagging & Syntactic Chunking | Char-CNN + Word Embedding Fusion + BiLSTM + Linear-Chain CRF | UDPOS (17 UPOS tags) | Character morphology extraction, Exact CRF Forward NLL Loss, Viterbi Dynamic Programming Decoding, Transition Matrix Heatmap, TPU/XLA support |
| [`05_question_answering_bidaf.ipynb`](notebooks/text/05_question_answering_bidaf.ipynb) | Extractive Question Answering (Reading Comprehension) | Bidirectional Attention Flow (BiDAF) + Highway Networks | SQuAD v1.1 | Context-to-Query (C2Q) & Query-to-Context (Q2C) Attention Flow, Start/End Span Cross-Entropy, Exact Match (EM) & Macro-F1, TPU/XLA support |
| [`06_unsupervised_contrastive_simcse.ipynb`](notebooks/text/06_unsupervised_contrastive_simcse.ipynb) | Self-Supervised Semantic Sentence Representations | SimCSE Transformer Encoder + InfoNCE / NT-Xent Contrastive Loss | WikiText-2 (Unlabeled) | Dropout as minimal data augmentation, In-Batch Negatives, Alignment & Uniformity metrics, 2D Latent t-SNE / PCA, Semantic Search, TPU/XLA support |

---

## Directory Structure

```text
pytorch-mnist-pipeline/
├── notebooks/
│   ├── audio/                          # Speech & audio processing suite (01 - 04)
│   │   ├── 01_speech_commands_convnext.ipynb
│   │   ├── 02_librispeech_conformer_ctc.ipynb
│   │   ├── 03_speech_denoising_conv_tasnet.ipynb
│   │   └── 04_speaker_verification_ecapa_tdnn.ipynb
│   ├── text/                           # Natural language processing suite (01 - 06)
│   │   ├── 01_news_classification_textcnn_attention.ipynb
│   │   ├── 02_causal_language_model_transformer.ipynb
│   │   ├── 03_machine_translation_seq2seq_transformer.ipynb
│   │   ├── 04_sequence_tagging_bilstm_crf.ipynb
│   │   ├── 05_question_answering_bidaf.ipynb
│   │   └── 06_unsupervised_contrastive_simcse.ipynb
│   └── vision/                         # Computer vision suite (00 - 09)
│       ├── 00_mnist_classification_resnet.ipynb
│       ├── 01_image_classification_eurosat_convnext.ipynb
│       ├── 02_object_detection_voc_retinanet.ipynb
│       ├── 03_semantic_segmentation_voc_deeplabv3.ipynb
│       ├── 04_optical_flow_flyingchairs_flownet.ipynb
│       ├── 05_stereo_matching_sceneflow_stereonet.ipynb
│       ├── 06_image_pairs_lfw_siamese.ipynb
│       ├── 07_image_captioning_coco_transformer.ipynb
│       ├── 08_video_classification_ucf101_r2plus1d.ipynb
│       └── 09_video_prediction_movingmnist_convlstm.ipynb
├── pyproject.toml                      # Project dependencies and environment specification
└── README.md
```

---

## Getting Started

### Prerequisites & Installation

Manage dependencies using [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`:

```bash
# Using uv (fastest)
uv sync

# Or using pip
pip install -r <(uv pip compile pyproject.toml)
```

### Running the Notebooks

Launch your preferred notebook interface from the repository root:

```bash
# Launch JupyterLab
jupyter lab

# Or launch Jupyter Notebook
jupyter notebook
```

Navigate to `notebooks/vision/`, `notebooks/audio/`, or `notebooks/text/` and run the desired notebook sequentially. Datasets will automatically be downloaded into the root `data/` directory upon first execution.

---

## Telemetry & Logging

All notebooks log scalar curves, metrics, and diagnostics to `runs/<domain>_<model>/<timestamp>`:

```bash
# Launch TensorBoard dashboard
tensorboard --logdir runs
```

- **Scalar Tag Scheme**: Standardized `Metric/Split` convention (`Loss/Train`, `Loss/Val`, `Accuracy/Train`, `Accuracy/Val`, `Metrics/CER`, `Metrics/Val_SDR_Gain_dB`).
- **Run Isolation**: Every training run generates a unique timestamped subfolder (`YYYY-MM-DD_HH-MM-SS`) preventing curve collision across repeated runs.
- **Nested Progress Display**: Outer epoch progress and inner batch progress bars are synchronized with custom `TQDM_BAR_FORMAT` formatting and non-destructive `tqdm.write()` checkpoint notifications.
