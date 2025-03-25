
# GEXSent: Gated Experts for Robust Sentiment Analysis Across Modalities

## Overview

GEXSent is a novel framework for **Multimodal Sentiment Analysis (MSA)** that leverages advanced models like **Modern BERT** for text and **CLIP** for visual feature extraction. It integrates two powerful feature enhancement modules: **Hierarchical Gated Mixture of Experts (H-GMoE)** and **Gated Attention Mechanism (GAM)**, enabling more focused and robust sentiment analysis across various modalities, including text, images, and videos.

The framework aims to address the challenges of traditional sentiment analysis methods by incorporating deep learning techniques to interpret complex emotional cues across multiple data modalities. GEXSent outperforms current state-of-the-art models on benchmark datasets such as **CMU-MOSI**, **CMU-MOSEI**, and **Memotion**, showing superior performance in fine-grained sentiment classification and sentiment intensity prediction.

The key components of the GEXSent framework include:
- **Unimodal Representation Learning**: Utilizes CLIP for visual feature extraction and Modern BERT for text feature extraction.
- **Hierarchical Gated Mixture of Experts (H-GMoE)**: Selectively emphasizes the most informative features from both text and visual modalities.
- **Gated Attention Mechanism (GAM)**: Focuses on the most relevant features for improved cross-modal interactions.

GEXSent has shown significant improvement in sentiment classification tasks by effectively handling complex multimodal inputs, demonstrating its ability to enhance sentiment understanding in real-world applications like social media analysis, entertainment, and customer feedback analysis.

## Installation
pip install -r requirements.txt


## Preprocessing and Feature Extraction

### Running `preprocess.py`

Before training, you need to preprocess the data. This step involves extracting pickle files required for training the model.

1. Open `preprocess.py` and change the required file paths for your dataset.
2. Run the script to extract features:

```bash
python preprocess.py
```

Ensure that you specify the correct paths to your input and output directories as per your environment.

## Training and Evaluation

### Running `main.py`

Once preprocessing is complete, you can proceed with training the model and evaluating it on the test set.

1. Modify the `config.py` file to adjust training parameters such as learning rate, batch size, epochs, etc. to suit your requirements.
2. Run the `main.py` script to begin training:

```bash
python main.py
```

After training, the model will automatically be evaluated on the test set and print the evaluation metrics.

### Notes

- The code is modular, and soon, updates will include training models like **Memtoion** and **MOSI** and will be released to enhance the framework.
- You may need to update paths and parameters as per your environment setup for correct execution.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, create a branch, and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
