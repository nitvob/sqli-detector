# SQL Injection Detection Using Fine-Tuned BERT

## Overview

This project aims to address the growing threat of SQL injection attacks by leveraging modern natural language processing (NLP) techniques. By fine-tuning the pre-trained BERT (Bidirectional Encoder Representations from Transformers) model, we develop an advanced system capable of identifying SQL injection attacks with high precision and recall. The model is trained on a labeled dataset of SQL queries and optimized using a triplet loss function to enhance its ability to differentiate malicious queries from benign ones.

## Key Features

- **Transformer-Based Approach**: Fine-tuning BERT for SQL injection detection.
- **Triplet Loss Optimization**: Enhancing the model's ability to discriminate between similar and dissimilar queries.
- **Baseline Comparison**: Evaluation of embeddings from both fine-tuned and base BERT models.
- **Logistic Regression Benchmark**: Using embeddings for binary classification with logistic regression.
- **Visualization**: Employing t-SNE to visualize the clustering of embeddings pre- and post-fine-tuning.

## Dataset

The dataset used in this project was sourced from Kaggle and contains:

- **SQL Queries**: Labeled as either benign (0) or malicious (1).
- **Rich Variety**: Covers a wide range of SQL commands for robust model training.

The dataset can be accessed [here](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset/data).

## Tools & Libraries

- **Transformers**: For accessing BERT models and NLP utilities.
- **Scikit-learn**: For implementing logistic regression and evaluation.
- **Pandas & NumPy**: For data preprocessing and numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **PyTorch**: For fine-tuning BERT and building data loaders.

## Experimental Procedure

1. **Data Preprocessing**:

   - Tokenization and padding using BERT tokenizer.
   - Conversion of SQL queries into input IDs and attention masks.
   - Splitting the dataset into training and validation sets.

2. **Baseline Evaluation**:

   - Use base BERT embeddings for logistic regression classification.
   - Evaluate using metrics like accuracy, precision, recall, and F1 score.

3. **Fine-Tuning BERT**:

   - Implement triplet loss for optimization.
   - Fine-tune BERT to improve embeddings' discrimination capability.

4. **Model Evaluation**:

   - Train logistic regression using fine-tuned embeddings.
   - Compare performance with baseline embeddings.

5. **Visualization**:
   - Apply t-SNE to visualize embedding clusters for benign and malicious queries.

## Results

### Key Performance Metrics

| Metric        | Base BERT Embeddings | Fine-Tuned BERT Embeddings |
| ------------- | -------------------- | -------------------------- |
| **Accuracy**  | 87.5%                | 99.8%                      |
| **Precision** | 84.0%                | 99.9%                      |
| **Recall**    | 84.4%                | 99.6%                      |
| **F1 Score**  | 84.2%                | 99.7%                      |

### Observations

- Fine-tuned BERT significantly outperformed base BERT in all metrics.
- The model achieved near-perfect accuracy and F1 score.
- t-SNE visualizations show clear separation between benign and malicious queries after fine-tuning.

## Conclusion

This project demonstrates the effectiveness of fine-tuning BERT for SQL injection detection. By leveraging triplet loss, the model produces embeddings that accurately differentiate between malicious and benign SQL queries. Future work could focus on:

- Exploring more efficient transformer architectures.
- Incorporating hybrid rule-based and machine-learning approaches.
- Scaling to larger datasets using cloud computing resources like AWS.

## How to Run the Project

1. **Environment Setup**:

   - Create a virtual environment: `python -m venv env` and activate it.
   - Install dependencies explicitly: `pip install torch transformers pandas numpy matplotlib seaborn scikit-learn`.

2. **Dataset Preparation**:

   - The dataset is included in the repository.

3. **Run the Code**:

   - Use the provided Jupyter notebook (`SQL_Injection_Detection_BERT.ipynb`) for step-by-step execution.

4. **Results Visualization**:
   - Generate visualizations for embedding clustering using t-SNE.

## Acknowledgments

- **Dataset**: Abu Syeed Sajid Ahmed ([Kaggle](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset/data)).
- **BERT Model**: Hugging Face Transformers.
- **References**: Research papers and techniques cited in the accompanying report.

For more details, refer to the project report.
