# Project Delivery Document

##  Project Overview

This project is a comprehensive text analysis system, comprising two core modules: **supervised classification** and **unsupervised clustering**. The system is capable of automatically identifying and distinguishing two categories of text — **Chinese web novels (Xianxia genre)** and **Western science fiction** — while also supporting unsupervised topic discovery, keyword extraction, and sentiment analysis. The system employs advanced natural language processing techniques to achieve high-precision classification and in-depth content analysis.

### Core Functionality

#### Supervised Classification Module
-  **Automatic text classification**: Given any text snippet, automatically determine whether it belongs to "Chinese web novel" or "Western science fiction"
-  **High-precision recognition**: Model accuracy reaches 99.99%, with an F1-Score of 99.99%
-  **Batch processing**: Supports single-text, batch-file, and interactive prediction
-  **Confidence assessment**: Provides prediction confidence to facilitate quality evaluation

#### Unsupervised Clustering Module
-  **Topic discovery**: Performs unsupervised topic modeling based on BERTopic to automatically discover thematic patterns in text
-  **c-TF-IDF keyword extraction**: Uses the class-level TF-IDF (c-TF-IDF) algorithm to extract topic keywords and compute importance scores
-  **Topic–sentiment mapping**: Computes a sentiment-weighted mean score for each topic, enabling correlation analysis between topics and sentiment
-  **Visualization analysis**: Generates a variety of visualization charts, including topic distribution charts and sentiment–topic quadrant charts

### Application Scenarios

- **Content moderation and classification**: Automatically identify text categories to support content review
- **Automatic labeling of text corpora**: Process texts in bulk and automatically generate category labels
- **Content recommendation systems**: Recommend content based on topic and sentiment
- **Topic discovery and analysis**: Discover thematic patterns in text in an unsupervised manner, and analyze topic distribution and sentiment tendencies
- **Academic research and analysis**: Support research tasks such as text mining, topic modeling, and sentiment analysis

---

##  Technical Approach

### I. Supervised Classification Module

#### Model Architecture

- **Base model**: BERT-base-uncased (12-layer Transformer, 110M parameters)
- **Task type**: Supervised binary classification
- **Input format**: Text sequence (maximum 512 tokens)
- **Output format**: Class label + confidence probability

### Technical Features

1. **Handling Data Imbalance**
   - Adopts a class-weighted loss function (Class Weighted Loss)
   - Automatically balances the learning weights of majority and minority classes
   - Ensures that both classes achieve good classification performance

2. **Stratified Sampling Strategy**
   - The training set and validation set use stratified sampling
   - Maintains consistent class distribution across training / validation sets
   - Ensures reliable evaluation results

3. **Complete Evaluation System**
   - Multi-dimensional metrics: accuracy, precision, recall, F1-Score
   - ROC curve and PR curve analysis
   - Confusion matrix visualization
   - Error sample analysis

### II. Unsupervised Clustering Module

#### Technical Architecture

- **Topic modeling framework**: BERTopic (unsupervised topic modeling based on BERT embeddings)
- **Embedding model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Clustering algorithm**: UMAP dimensionality reduction + HDBSCAN clustering
- **Keyword extraction**: c-TF-IDF (class-level TF-IDF)
- **Sentiment analysis model**: RoBERTa-base-sentiment

#### Core Algorithms

1. **Unsupervised Topic Clustering**
   - **Text embedding**: Use SentenceTransformer to convert text into high-dimensional vector representations
   - **Dimensionality reduction**: Use UMAP to reduce high-dimensional embeddings to a low-dimensional space (default 2 dimensions)
   - **Density clustering**: Use HDBSCAN for density-based clustering, automatically discovering topic clusters
   - **Noise handling**: Documents that cannot be assigned are marked as noise (topic_id = -1)

2. **c-TF-IDF Keyword Extraction**
   - **Algorithm principle**: c-TF-IDF (class-based TF-IDF) is a variant of TF-IDF specifically designed for topic modeling
   - **Formula**:
     ```
     c-TF-IDF(t, k) = TF(t, k) × log(1 + N / DF(t))
     ```
     where:
     - `TF(t, k)`: term frequency of word t within topic k
     - `N`: total number of topics
     - `DF(t)`: number of topics containing word t
   - **Advantage**: Compared with conventional TF-IDF, c-TF-IDF can better identify keywords specific to a topic
   - **Output**: A keyword list and its c-TF-IDF scores for each topic; higher scores indicate the word is more important to that topic

3. **Topic–Sentiment Weighted Mapping**
   - **Sentiment computation**: Use the RoBERTa-base-sentiment model to compute the sentiment polarity score for each text segment
     - Sentiment score range: [-1, 1]
     - -1: extremely negative, 0: neutral, 1: extremely positive
     - Formula: `sentiment_polarity = Prob(Positive) - Prob(Negative)`
   
   - **Weighted-average computation**:
     - If topic probabilities (topic_probability) are available, use the weighted average:
       ```
       E_k = Σ(p_i × S_i) / Σ(p_i)
       ```
       where:
       - `E_k`: the mean sentiment score of topic k
       - `p_i`: the probability that document i belongs to topic k
       - `S_i`: the sentiment score of document i
     
     - If no topic probabilities are available, use a simple average:
       ```
       E_k = Mean(S_i)
       ```
   
   - **Output metrics**:
     - `Sentiment_Mean`: mean topic sentiment score
     - `Sentiment_Std`: standard deviation of sentiment scores
     - `Frequency`: number of documents in the topic
     - `Positive_Count`: number of positive texts
     - `Negative_Count`: number of negative texts
     - `Neutral_Count`: number of neutral texts

#### Technical Features

1. **No Labeled Data Required**
   - Fully unsupervised; no pre-labeled training data is needed
   - Automatically discovers thematic patterns in the data

2. **Strong Interpretability**
   - Each topic comes with a keyword list and c-TF-IDF scores
   - Visualizes topic distribution and keyword importance

3. **Supports Multi-Dimensional Analysis**
   - Topic frequency statistics
   - Topic source distribution (Chinese vs. Western comparison)
   - Topic sentiment-tendency analysis
   - Topic–sentiment quadrant chart visualization

4. **Flexible Stopword Handling**
   - 200+ built-in stopwords (including proper nouns, common function words, etc.)
   - Supports custom stopword lists
   - Stopword filtering implemented via CountVectorizer

#### Performance Metrics

Based on the topic modeling results over 436,934 documents:

- **Number of topics discovered**: 158 valid topics
- **Number of documents in valid topics**: 381,245 (87.25%)
- **Number of noise documents**: 55,689 (12.75%)
- **Topic coverage**: The top 5 topics account for approximately 26.50% of documents
- **Topic distinctiveness**: Can clearly distinguish different types of content (military, alchemy, agriculture, technology, witches, dragons, etc.)

**Characteristics of Topic Distribution**:
- More balanced distribution: The top 5 topics account for approximately 26.50% of documents
  - Topic 0 (experts / space / skeleton): 26,444 documents (6.05%)
  - Topic 1 (thinking / poison / paintings): 26,413 documents (6.05%)
  - Topic 2 (refinement / heavenly dao / destiny): 23,117 documents (5.29%)
  - Topic 3 (situ / spell / killing intent): 21,500 documents (4.92%)
  - Topic 4 (mysterious / possession / poison): 18,292 documents (4.19%)
- Long-tail distribution: Most topics have a very small share (<1%), with 158 valid topics in total
- Diverse sources: The updated 200 Western science fiction books make the topic sources more diverse

**c-TF-IDF Keyword Quality**:
- Good semantic relevance: Topic keywords exhibit clear semantic relatedness
  - For example, Topic 7 (medicine theme): medicine(0.3165), scarlet(0.3067), lychee(0.2528)
  - For example, Topic 8 (character / plot theme): character(0.3418), storyline(0.3203), intelligence(0.2319)
- High distinctiveness: Keywords from different topics effectively distinguish them
  - Medicine theme (Topic 7): medicine, scarlet, lychee, daoist
  - Criminal-psychology theme (Topic 22): homicide, psychology, victims
  - Transportation / travel theme (Topic 20): car, train, drove
- Strong interpretability: The keywords of each topic clearly convey the topic's meaning, and c-TF-IDF scores intuitively reflect keyword importance

**Topic–Sentiment Mapping Results**:
- Sentiment score range: [-1, 1]; sentiment-weighted mean scores have been computed for 158 topics
- Sentiment distribution: All 158 topics have a negative sentiment (consistent with the conflict-driven nature of novel plots)
- Sentiment range: from -0.439 (Topic 137) to -0.030 (Topic 88)
- Examples of negative topics: Topic 137 (the most negative, -0.439), Topic 44 (media / surveillance, -0.388), Topic 26 (-0.379)

---

##  Dataset

### Data Sources

- **Chinese web novels (Chinese_Xianxia)**: 20 Chinese Xianxia / cultivation novels
- **Western science fiction (Western_SciFi)**: 200 Western classic science fiction novels

### Data Statistics

| Category | Number of Samples | Proportion | Average Characters | Average Tokens |
|----------|------------------|------------|-------------------|----------------|
| Chinese_Xianxia | 299,289 | 68.5% | ~1,146 | ~287 |
| Western_SciFi | 137,645 | 31.5% | ~1,148 | ~287 |
| **Total** | **436,934** | **100%** | - | - |

**Class ratio**: 2.17:1 (moderate imbalance)

### Data Preprocessing

1. **Text Cleaning**
   - Remove special characters and formatting markers
   - Unify line-break handling
   - Retain core text content

2. **Text Segmentation**
   - Sliding-window segmentation (window size: 1200 characters, overlap: 150 characters)
   - Ensures each segment fits within BERT's 512-token limit
   - Minimum segment length: 200 characters

3. **Data Balancing**
   - Use a class-weighted loss function during training
   - Retain all data — no information is lost
   - Automatically balances the learning process through weights

---

##  Model Performance

### Final Model Metrics (Stage 2 — Class Weighting + Focal Loss)

Evaluation results on the validation set (87,387 samples):

#### Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **99.99%** |
| **Macro-Averaged F1-Score** | **99.99%** |
| **Weighted-Averaged F1-Score** | **99.99%** |
| **ROC-AUC** | **1.0000** |
| **PR-AUC** | **1.0000** |

#### Detailed Per-Class Metrics

**Western_SciFi (Western Science Fiction)**
- Precision: **100.00%**
- Recall: **99.99%**
- F1-Score: **99.99%**
- Support: 27,529

**Chinese_Xianxia (Chinese Web Novels)**
- Precision: **100.00%**
- Recall: **99.99%**
- F1-Score: **99.99%**
- Support: 59,858

#### Confusion Matrix

| True \\ Predicted | Western_SciFi | Chinese_Xianxia |
|-------------------|---------------|-----------------|
| Western_SciFi | 27,526 | 3 (0.01%) |
| Chinese_Xianxia | 6 (0.01%) | 59,852 |

**Key Observations**:
- 3 Western science fiction samples were misclassified as Chinese web novels, and 6 Chinese web novel samples were misclassified as Western science fiction
- Total misclassifications: 9 (misclassification rate of only 0.0103%)
- F1-Score difference: 0.0000 (well below the ideal threshold of 0.1)

#### Prediction Confidence

- **Average confidence**: 99.90%
- **Low-confidence samples (<0.7)**: 0 (0.00%)
- **High-confidence samples (≥0.9)**: 87,382 (99.99%)

### Training Configuration

- **Model**: BERT-base-uncased (local version)
- **Batch size**: 16
- **Learning rate**: 2e-5
- **Training epochs**: 3 epochs
- **Training samples**: 349,547
- **Validation samples**: 87,387
- **Training mode**: Stage 2 (Class Weighting + Focal Loss)

---

##  Project File Structure

```
clean/
├── 1.py                              # Script file
├── 2.py                              # Script file
├── 3.py                              # Data segmentation / training-set generation
├── 4.py                              # Script file
├── Cleaned_Chinese_Trans_V3/         # Preliminarily cleaned Chinese web novel texts (intermediate output)
├── Cleaned_English_V2/               # Preliminarily cleaned Western SciFi texts (intermediate output)
├── Deep_Cleaned_Chinese/             # Deeply cleaned Chinese texts
├── Deep_Cleaned_English/             # Deeply cleaned English texts
├── __pycache__/                      # Python compilation cache directory
├── add_timestamps.py                 # Timestamp processing script
├── all-MiniLM-L6-v2-local/           # Local SentenceTransformer model
├── bert-base-uncased-local/          # Local BERT-base model
├── bert_training_dataset.csv         # Supervised classification training dataset
├── bertopic_analysis/                # Unsupervised BERTopic analysis module
│   ├── __pycache__/
│   ├── requirements_bertopic.txt     # Dependency list for the BERTopic module
│   ├── sentiment_analysis.py         # Topic sentiment analysis script
│   ├── sentiment_quadrant_chart.html # Sentiment–topic quadrant chart
│   ├── stopwords.py                  # Stopwords configuration
│   ├── topic_analysis.csv            # Topic clustering summary results
│   ├── topic_analysis_detailed_topics.csv
│   ├── topic_analysis_document_topics.csv
│   ├── topic_analysis_output/        # Topic visualization output directory
│   ├── topic_modeling.py             # BERTopic topic modeling script
│   ├── topic_sentiment_final.csv     # Final topic-sentiment statistics table
│   └── visualize_topic_comparison.py # Topic comparison and visualization script
├── check_clean.py                    # Data-cleaning inspection tool
├── check_gpu.py                      # GPU environment check tool
├── clean_check_report.csv            # Data-cleaning inspection report
├── deep_clean_report.csv             # Deep-cleaning report
├── deleted_report_cn_v3.csv          # Chinese deletion-record report
├── deleted_report_v2.csv             # English deletion-record report
├── download_model_local.py           # Script to download / prepare the local model
├── download_roberta_sentiment_model.py # Script to download the RoBERTa sentiment model
├── evaluate_saved_models.py          # Script to evaluate saved models
├── predict.py                        # Prediction script (single / batch / interactive)
├── replot_training_curves.py         # Script to redraw training curves
├── requirements.txt                  # Main dependency list
├── roberta-base-sentiment-local/     # Local RoBERTa sentiment model
├── train_bert.py                     # BERT training script
├── training_output_stage2_classweights_focal/  # Stage 2 training output directory
│   ├── models/
│   │   └── best_model_epoch_3.pt
│   └── reports/
│       ├── classification_report_stage2_classweights_focal.txt
│       ├── confusion_matrix_stage2_classweights_focal.png
│       ├── roc_pr_curves_stage2_classweights_focal.png
│       ├── training_curves_stage2_classweights_focal.png
│       ├── metrics_stage2_classweights_focal.json
│       └── training_config_stage2_classweights_focal.json
├── Stopwords_Description.md          # Stopwords-related documentation
├── Data_Description_Document.md      # Detailed dataset documentation
├── Final_Erroneous_Stopwords_List.md # Final list of erroneous stopwords
├── 英文中方/                          # English corpus (Chinese side)
├── 英文西方/                          # English corpus (Western side)
├── 获奖(完结)时间/                    # Data on award-winning / completion times
├── Detailed_Data_Results_Report.md   # Detailed data results analysis report
└── Project_Delivery_Document.md      # This delivery document
```

---

##  Usage Guide

### Environment Requirements

- **Python version**: 3.10 or above (3.10–3.12 recommended)
- **Operating system**: Linux / Windows / macOS
- **GPU**: GPU acceleration is recommended (CUDA support); CPU can also run, but is very slow (RTX 4090D recommended)
- **VRAM**: 24 GB recommended

### Installation Steps

1. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

Dependency list (main environment, see `requirements.txt`):
- torch >= 2.0.0
- transformers >= 4.41.0, < 6.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tqdm >= 4.62.0
 - bertopic >= 0.15.0
 - sentence-transformers >= 5.0.0
 - plotly >= 5.0.0

If the unsupervised clustering module needs to run in a **separate environment**, refer to `bertopic_analysis/requirements_bertopic.txt`, which additionally lists:
- umap-learn >= 0.5.0
- hdbscan >= 0.8.0
- scipy >= 1.7.0

2. **Prepare the BERT model**

The project already includes the local BERT model files (`bert-base-uncased-local/`); no additional download is required.

**Important Notes**:
- Ensure that the `bert-base-uncased-local/` directory is located in the project root
- If migrating from a server to a local machine, the paths in the model configuration will automatically adapt
- The scripts automatically locate the local model files — no manual configuration changes are needed

### Usage

#### 1. Single-Text Prediction

```bash
python predict.py --text "Your text here"
```

**Example**:
```bash
python predict.py --text "The starship drifted through the void, its engines silent as the crew prepared for the jump to hyperspace."
```


#### 2. Batch-File Prediction

```bash
python predict.py --file input.txt --output results.csv
```

`input.txt` format: one text per line

**Output**: `results.csv` contains the predicted category and confidence

#### 3. Interactive Prediction

```bash
python predict.py --interactive
```

Enters interactive mode, where you can continuously input texts for prediction. Enter `quit` or `exit` to exit.

#### 4. Specifying a Model Path

```bash
python predict.py --model path/to/model.pt --text "Your text"
```

By default, the best model is used: `training_output_stage2_classweights_focal/models/best_model_epoch_3.pt`

### Unsupervised Topic Modeling (Newly Added Feature)

#### 1. Topic Modeling

```bash
cd bertopic_analysis
python topic_modeling.py --input_csv ../bert_training_dataset.csv --output topic_analysis.csv
```

**Parameter Description**:
- `--input_csv`: Input CSV file (must contain a `text` column)
- `--output`: Output CSV file path
- `--num_topics`: Specify the number of topics (optional; auto-determined by default)
- `--min_topic_size`: Minimum topic size (default 10)
- `--output_dir`: Visualization output directory (default `topic_analysis_output`)

**Output Files**:
- `topic_analysis.csv`: Topic frequency statistics table (contains topic ID, keywords, frequency, source distribution, etc.)
- `topic_analysis_detailed_topics.csv`: Detailed topic information
- `topic_analysis_document_topics.csv`: Topic assignment for each document
- `topic_analysis_output/`: Visualization charts (HTML format)

#### 2. Topic–Sentiment Mapping Analysis

```bash
cd bertopic_analysis
python sentiment_analysis.py
```

**Functionality**:
- Reads `topic_analysis_document_topics.csv`
- Computes a sentiment score for each document
- Computes the sentiment-weighted mean score for each topic
- Generates a sentiment–topic quadrant chart

**Output Files**:
- `topic_sentiment_final.csv`: Topic-sentiment statistics table
- `sentiment_quadrant_chart.html`: Interactive sentiment–topic quadrant chart

**Quadrant Chart Description**:
- X-axis: Topic popularity (number of documents)
- Y-axis: Mean topic sentiment score (-1 to 1)
- Point size: Represents topic frequency
- Point color: Represents sentiment tendency (red = positive, blue = negative)

### Training a New Model (Optional)

If you need to retrain or adjust the model:

#### Stage 1: Quick Verification (Downsampling)

```bash
python train_bert.py --mode stage1
```

- Uses downsampling to balance the data
- Quick verification of model feasibility
- Suitable for rapid testing

#### Stage 2: Optimized Model (Class Weighting)

```bash
python train_bert.py --mode stage2
```

- Uses a class-weighted loss function
- Retains all data
- Achieves the best performance (recommended)

#### Custom Parameters

```bash
python train_bert.py --mode stage2 \
    --batch_size 32 \
    --epochs 5 \
    --lr 3e-5
```

---

##  Technical Details

### Unsupervised Clustering Technical Details

#### Detailed Explanation of the c-TF-IDF Algorithm

c-TF-IDF (class-based TF-IDF) is the core algorithm of BERTopic, used to extract topic keywords:

1. **Differences from Conventional TF-IDF**:
   - Conventional TF-IDF: Computes the importance of a word across the entire document collection
   - c-TF-IDF: Computes the importance of a word within a specific topic (class)
   - Advantage: Better at identifying keywords specific to a topic and reducing interference from generic words

2. **Computation Workflow**:
   ```
   For each topic k:
     1. Merge all documents belonging to topic k into a single "class document"
     2. Compute the term frequency (TF) of each word within this class document
     3. Compute the topic-distribution frequency (DF) of each word
     4. Compute the c-TF-IDF score = TF × log(1 + N/DF)
   ```

3. **Keyword Ranking**:
   - Sorted in descending order of c-TF-IDF score
   - The higher the score, the more important the word is to the topic
   - Typically the top 10–15 words are taken as the topic's keywords

#### Topic–Sentiment Weighted Mapping Computation

The mean sentiment score for a topic takes into account the probability that documents belong to the topic:

1. **Weighted-Average Formula** (recommended):
   ```
   E_k = Σ(p_i × S_i) / Σ(p_i)
   ```
   where:
   - `E_k`: the mean sentiment score of topic k
   - `p_i`: the probability that document i belongs to topic k (obtained from HDBSCAN clustering)
   - `S_i`: the sentiment score of document i (between -1 and 1)

2. **Simple-Average Formula** (when no probability information is available):
   ```
   E_k = Mean(S_i)
   ```

3. **Advantages**:
   - The weighted average better reflects the true sentiment tendency of the topic
   - High-probability documents have a greater impact on topic sentiment
   - Reduces the influence of noise documents on the results

#### Clustering Parameter Tuning

1. **min_topic_size** (Minimum Topic Size):
   - Default value: 10
   - Function: Controls the minimum number of documents per topic
   - Tuning suggestions:
     - Large data volume: can be increased moderately (15–20)
     - Small data volume: can be decreased moderately (5–8)
     - Too many noise documents: decrease this value

2. **nr_topics** (Number of Topics):
   - Default value: None (auto-determined)
   - Function: Limits the number of topics by merging similar topics
   - Tuning suggestions:
     - Too many topics: set to 20–30
     - Too few topics: do not set; let the algorithm discover them automatically

3. **embedding_model** (Embedding Model):
   - Recommended: all-MiniLM-L6-v2 (fast and effective)
   - Alternative: BERT-base-uncased (more accurate but slower)
   - Choice criteria: data volume, computational resources, precision requirements

### Supervised Classification Technical Details

### Data Imbalance Handling Strategy

The project adopts a **class-weighted loss function** to handle the data imbalance problem:

1. **Automatic Weight Computation**
   - Weights are automatically computed based on the sample counts of each class in the training set
   - The minority class (Western_SciFi) receives a higher weight (around 1.59)
   - The majority class (Chinese_Xianxia) receives a weight of around 0.73

2. **Advantages**
   - Retains all data — no information is lost
   - Automatically balances the learning process through loss weights
   - Suitable for production environments; the model is able to learn more features

3. **Effectiveness Validation**
   - The F1-Scores of the two classes are very close (difference of only 0.0000)
   - Minority-class recall reaches 99.99%
   - Overall accuracy reaches 99.99%

### Model Training Workflow

1. **Data Loading**: Load preprocessed data from a CSV file
2. **Data Splitting**: 80% training set, 20% validation set (stratified sampling)
3. **Model Initialization**: Load the pre-trained BERT-base model
4. **Training Loop**:
   - Record training loss and validation loss for each epoch
   - Compute F1-Score, precision, recall, and other metrics
   - Save the best model (based on validation loss and F1-Score)
5. **Evaluation and Reporting**:
   - Generate a detailed classification report
   - Draw the confusion matrix, ROC curve, and PR curve
   - Save error sample analysis

### Description of Evaluation Metrics

- **Accuracy**: The proportion of correctly predicted samples
- **Precision**: Among samples predicted as positive, the proportion that are truly positive
- **Recall**: Among samples that are truly positive, the proportion correctly predicted
- **F1-Score**: The harmonic mean of precision and recall — a composite metric
- **ROC-AUC**: The area under the ROC curve — measures the overall performance of the classifier
- **PR-AUC**: The area under the precision–recall curve — more meaningful for imbalanced data

---

##  Performance Analysis

### Training Curve Analysis

The model training process was stable, and all metrics continually improved:

1. **Loss curves**: Both training loss and validation loss continually decreased, with no sign of overfitting
2. **F1-Score curves**: The F1-Scores of the two classes gradually converged during training
3. **Class balance**: The F1 difference decreased to 0.0000, reaching an ideal state

### Error Sample Analysis

- **Total number of errors**: 9 (only 0.0103% of the validation set)
- **Error types**: 3 Western_SciFi misclassified as Chinese_Xianxia; 6 Chinese_Xianxia misclassified as Western_SciFi
- **Possible causes**: Blurred text-feature boundaries, or content with mixed styles

### Model Advantages

1. **High precision**: 99.99% accuracy, meeting production-environment requirements
2. **Class balance**: F1-Scores of both classes are close — no bias
3. **High confidence**: 99.99% of samples have confidence ≥0.9 — reliable predictions
4. **Strong generalization**: Excellent performance on the validation set; good generalization capability

---

##  GPU Usage Guide

### Runtime Environment (AutoDL Cloud GPU Server)

- **Image environment**:
  - PyTorch 2.5.1
  - Python 3.12 (ubuntu22.04)
  - CUDA 12.4

- **Hardware configuration**:
  - **Memory**: 80 GB
  - **Storage**:
    - System disk: 30 GB
    - Data disk (free): 50 GB

### Checking GPU Usage

Run check_gpu.py

---

##  Output File Description

### Model Files

- `best_model_epoch_X.pt`: Contains the model state, optimizer state, epoch information, and training configuration

### Evaluation Reports

- `classification_report_*.txt`: Detailed text classification report
- `metrics_*.json`: All metrics in JSON format
- `training_config_*.json`: Training configuration parameters

### Visualization Files

- `confusion_matrix_*.png`: Confusion matrix (with percentage annotations)
- `roc_pr_curves_*.png`: ROC curve and PR curve
- `training_curves_*.png`: Multi-dimensional training curves (loss, F1, precision/recall, class balance)

### Error Analysis

- `error_samples_*.csv`: All misclassified samples, containing the original text, true label, predicted label, confidence, etc.

---

##  Notes

### Data Quality

- Ensure that the input text has been properly cleaned and preprocessed
- Text length should be controlled between 200 and 2000 characters
- Avoid including excessive special characters or formatting markers

### Model Limitations

- **Maximum input length**: 512 tokens (about 2000 characters)
- **Language support**: Primarily targeted at English text; Chinese text requires corresponding adjustments
- **Class limitation**: Currently supports only binary classification (Chinese web novels vs. Western science fiction)

### Performance Optimization

- GPU acceleration is recommended for training and inference
- During batch prediction, you can adjust the `batch_size` to balance speed and memory usage
- For large-scale prediction tasks, consider using multiprocessing or distributed processing

### Model Updates

- If the data distribution changes, retraining the model is recommended
- Regularly evaluate model performance and monitor accuracy and error rates
- Collect error samples for model improvement

### Notes on Local Usage

**Path Issue Handling**:
- If migrating from a server to a local machine, the model configuration may contain absolute server paths
- The script will automatically try multiple paths to locate the BERT model files:
  1. The original path in the configuration
  2. The path relative to the script directory
  3. The `bert-base-uncased-local/` directory in the project root
- Make sure the `bert-base-uncased-local/` directory exists in the project root
- If you encounter path errors, check whether the directory structure is correct

**Common Error Resolution**:
- `OSError: Incorrect path_or_model_id`: Make sure the `bert-base-uncased-local/` directory exists
- `FileNotFoundError: Model file does not exist`: Check that the model file path is correct
- If the problem persists, manually specify the model path: `python predict.py --model <model_path> --text "..."`

**About FutureWarning Warnings**:
- At runtime, you may see the warning `FutureWarning: torch.utils._pytree._register_pytree_node is deprecated`
- **This is normal**: This is a transformers / PyTorch version compatibility notice and does not affect functionality
- **Cause**: transformers 4.36.2 uses a soon-to-be-deprecated PyTorch API, and newer PyTorch versions will warn about it
- **Handling**: The code already automatically suppresses these warnings; if you still see them, they can be ignored without affecting prediction results
- **Solution**: Updating the transformers library to a newer version in the future will eliminate the warning (the current version is already stable enough)

---

##  Project Summary

### Project Achievements

#### Supervised Classification Module
- **High-precision model**: Accuracy 99.99%, F1-Score 99.99%
- **Class balance**: F1-Score difference between the two classes is only 0.0000
- **Complete toolchain**: A full pipeline from data preprocessing to model training, evaluation, and prediction

#### Unsupervised Clustering Module
- **Topic discovery**: Automatically discovered 158 valid topics, covering 87.25% of the documents (381,245 documents)
- **c-TF-IDF keyword extraction**: Extracts highly relevant keywords for each topic and computes importance scores
- **Topic–sentiment mapping**: Implements correlation analysis between topics and sentiment, generating a sentiment–topic quadrant chart
- **Multi-dimensional visualization**: Topic distribution charts, keyword importance charts, sentiment quadrant charts, etc.

### Technical Highlights

#### Supervised Classification
1. **Data imbalance handling**: Combines class weighting and Focal Loss, effectively solving the 2.17:1 imbalance problem
2. **Complete evaluation system**: Multi-dimensional metrics, visualization analysis, and error sample analysis
3. **Production-ready**: Provides a complete prediction interface — supports single, batch, and interactive prediction
4. **Extensibility**: Clear code structure that is easy to extend and maintain

#### Unsupervised Clustering
1. **c-TF-IDF algorithm**: Uses class-level TF-IDF to extract topic keywords — better suited for topic modeling than conventional TF-IDF
2. **Topic–sentiment weighted mapping**: Considers the probability that documents belong to a topic, using a weighted average to compute the mean topic sentiment score
3. **Unsupervised learning**: Requires no labeled data; automatically discovers thematic patterns in the text
4. **Multi-dimensional analysis**: Supports analyses across multiple dimensions — topic frequency, source distribution, sentiment tendencies, etc.
5. **Rich visualization**: Interactive charts that support topic exploration and analysis

### Deliverables

#### Supervised Classification Module
- Trained model files (BERT classification model)
- Complete source code (training, prediction, and evaluation scripts)
- Detailed evaluation reports and visualizations
- User documentation and technical documentation
- Data preprocessing scripts

#### Unsupervised Clustering Module
- Topic modeling script (topic_modeling.py)
- Sentiment analysis script (sentiment_analysis.py)
- Topic analysis results (CSV format)
- Visualization charts (HTML format)
- Topic–sentiment mapping results
- Technical documentation and usage instructions

---