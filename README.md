# ABSA-model

Aspect-Based Sentiment Analysis (ABSA) for Vietnamese reviews using MinPhoBERT combined with Contrastive Learning.

![Overall Model Architecture](https://github.com/dluong1210/ABSA-model/blob/main/overall_model.png)

## 1. Installation

### 1.1. Install required libraries

Navigate to the `src` directory:

```sh
cd src
```

Run the installation script to install necessary libraries:

```sh
sh install.sh
```

### 1.2. Download dataset

The dataset used in this research is UIT-ViSFD. Run the following command to download the dataset from Kaggle:

```sh
python absa_download_data.py
```

**Note:** Update the dataset path in configuration files if needed, as the current path is hardcoded.

## 2. Train the model

Run the following command to train the model:

```sh
python main.py -epochs 10 -layer_bert 6 -cl_alpha 0.2 -log_path=path_here
```

Key parameters:

- `-epochs`: Number of training epochs (default is 10).
- `-layer_bert`: Number of layers in MinPhoBERT (default is 6 to reduce model complexity).
- `-cl_alpha`: Weight factor for Contrastive Learning.
- `-log_path`: Path to save log files during training and evaluation.

## 3. Model Architecture

The ABSA model consists of two main stages:

1. **Aspect Detection Model**: Identifies aspects in a sentence.
2. **Sentiment Classification Model**: Predicts sentiment (positive, negative, neutral) for each aspect.

MinPhoBERT serves as the backbone, combined with Contrastive Learning to enhance semantic representation.

## 4. Data Augmentation

As the UIT-ViSFD dataset is imbalanced, two augmentation methods are applied:

1. **Duplicate Data**: Duplicate sentences containing underrepresented aspects.
2. **Synthetic Data Generation**: Generate new data using the Gemma2:2b model.

## 5. Results

- **Aspect Detection** achieves **F1-score 0.87**, an improvement of **3.57%** over the baseline..
- **Sentiment Classification** achieves **F1-score 0.82**, an **improvement of 30.16%** over the baseline..
- Improvement over baseline models such as Bi-LSTM and PhoBERT.

## 6. Key Contributions

- Utilizing MinPhoBERT to optimize performance on Vietnamese data.
- Applying Contrastive Learning to enhance semantic representation.
- Addressing data imbalance using two augmentation strategies.
