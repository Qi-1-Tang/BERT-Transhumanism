# Data File Description Document

This document provides detailed descriptions of the purpose, format, field meanings, and usage methods of all data files in the project, helping users understand the role and content of each data file.

**Document Update Notes**:
- **Update Content**: Based on the updated corpus (20 Chinese web novels + 200 Western sci-fi novels), data cleaning, supervised classification, and unsupervised clustering analysis were re-performed.
- **Data Scale Changes**:
  - Total records: increased from 161,724 to 436,934
  - Chinese books: increased from 10 to 20
  - Western books: increased from 20 to 200
  - Class ratio: improved from imbalanced (12.5:1) to more balanced (2.17:1)

**Path Description**:
- File paths in this document start with `/root/autodl-tmp/clean/`, which is the actual path on the server.
- If used in a local environment, please adjust the paths accordingly.
- Relative paths: All paths are relative to the project root directory (`clean/`).

---

## Table of Contents

1. [Training Data Files](#1-training-data-files)
2. [Data Cleaning Report Files](#2-data-cleaning-report-files)
3. [Model Training Output Files](#3-model-training-output-files)
4. [Topic Analysis Data Files](#4-topic-analysis-data-files)
5. [Original Text Files](#5-original-text-files)
6. [Model Configuration Files](#6-model-configuration-files)
7. [Data File Usage Guide](#7-data-file-usage-guide)

---

## 1. Training Data Files

### 1.1 `bert_training_dataset.csv`

**File Path**: `/root/autodl-tmp/clean/bert_training_dataset.csv`


**File Purpose**:
- This is the main dataset for BERT model training
- Contains all preprocessed text chunks and their labels
- Used for training and validation of the supervised classification model

**File Format**: CSV (Comma-Separated Values)

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `text` | String | Text content (cleaned text chunk) | "The starship drifted through the void..." |
| `source` | String | Text source category | "Western_SciFi" or "Chinese_Xianxia" |
| `book_name` | String | Source book name | "Dune" or "Lord of the Mysteries" |
| `chunk_id` | Integer | Chunk number within the book | 1, 2, 3, ... |
| `char_count` | Integer | Character count of text | 1200 |
| `estimated_tokens` | Integer | Estimated token count (for BERT) | 300 |
| `Timestamps` | String | Timestamp corresponding to the text (book publication/completion/award year) | "2016" |

**Data Statistics**:
- **Total Records**: 436,934
- **Western_SciFi (Western Sci-Fi)**: 137,645 records (31.5%)
- **Chinese_Xianxia (Chinese Web Novels)**: 299,289 records (68.5%)
- **Average Character Count**: ~1,147 characters
- **Average Token Count**: ~287 tokens

**Data Source**:
- Derived from 220 novels (20 Chinese web novels + 200 Western sci-fi)
- Underwent text cleaning and chunking
- Used sliding window segmentation (window size 1200 characters, overlap 150 characters)


**Notes**:
- The file is large; chunked reading (`chunksize` parameter) is recommended
- Text has been cleaned, with junk content and formatting markers removed
- Each text chunk length is controlled within BERT's 512 token limit

---

## 2. Data Cleaning Report Files

### 2.1 `clean_check_report.csv`

**File Path**: `/root/autodl-tmp/clean/clean_check_report.csv`

**File Purpose**:
- Records all junk content detected during data cleaning
- Used for quality control and validation of cleaning effectiveness
- Helps understand the detailed process of data cleaning

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `文件夹` (Folder) | String | Folder where the file is located | "Cleaned_Chinese_Trans_V3" |
| `文件名` (File Name) | String | Source file name | "clean_My House of Horrors.txt" |
| `行号` (Line Number) | Integer | Line number where junk content is located | 14 |
| `内容` (Content) | String | Content marked as junk | "..." |
| `垃圾类型` (Junk Type) | String | Classification of junk content | "纯分割线" (Pure Separator) |

**Junk Type Description**:
- **Pure Separator (纯分割线)**: Lines containing only separators (e.g., "...", "---", etc.)
- **Translator Credits (翻译人员名单)**: Lines containing translator, editor, etc. information
- **NovelFull Watermark (NovelFull水印)**: Website watermark text
- **URL Deletion (URL删除)**: Lines containing URLs
- **Other (其他)**: Other types of junk content


---

### 2.2 `deleted_report_cn_v3.csv`

**File Path**: `/root/autodl-tmp/clean/deleted_report_cn_v3.csv`

**File Purpose**:
- Records content deleted during cleaning of Chinese web novels (Chinese_Xianxia)
- Detailed records of deletion reasons and locations

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `文件名` (File Name) | String | Source file name | "Library of Heaven is Path.txt" |
| `行号` (Line Number) | Integer | Line number where deleted content is located | 10 |
| `原内容` (Original Content) | String | Original deleted content | "Translator: StarveCleric Editor: Thaddpo" |
| `原因` (Reason) | String | Deletion reason | "翻译人员名单" (Translator Credits) |

**Deletion Reason Types**:
- **Translator Credits (翻译人员名单)**: Translator, editor information
- **NovelFull Watermark (NovelFull水印)**: Website watermark
- **URL Deletion (URL删除)**: Content containing URLs
- **Pure Separator (纯分割线)**: Lines with only separators
- **Other (其他)**: Other content needing deletion


---

### 2.3 `deleted_report_v2.csv`

**File Path**: `/root/autodl-tmp/clean/deleted_report_v2.csv`

**File Purpose**:
- Records content deleted during cleaning of Western sci-fi (Western_SciFi)
- Detailed records of deletion reasons and locations

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `文件名` (File Name) | String | Source file name | "Blindsight.txt" |
| `行号` (Line Number) | Integer | Line number where deleted content is located | 7574 |
| `内容` (Content) | String | Original deleted content | "www.feedbooks.com" |
| `原因` (Reason) | String | Deletion reason | "包含网址 (URL)" (Contains URL) |

**Deletion Reason Types**:
- **Contains URL (包含网址)**: Website links
- **Navigation (目录导航)**: Table of contents/index content
- **Highly Repeated Long Sentences (长句高频重复)**: Sentences appearing repeatedly (e.g., 26 times)
- **Other (其他)**: Other content needing deletion


---

### 2.4 `deep_clean_report.csv`

**File Path**: `/root/autodl-tmp/clean/deep_clean_report.csv`

**File Purpose**:
- Records all junk content detected during deep data cleaning
- Used for quality control and validation of cleaning effectiveness
- Contains more detailed cleaning information (for the updated corpus)

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `来源文件夹` (Source Folder) | String | Folder where the file is located | "Cleaned_English_V2" |
| `文件名` (File Name) | String | Source file name | "clean_Powers.txt" |
| `行号` (Line Number) | Integer | Line number where junk content is located | 561 |
| `原内容` (Original Content) | String | Original content marked as junk | "♦ ♦ ♦" |
| `原因` (Reason) | String | Classification of junk content | "纯符号/分割线" (Pure Symbols/Separator) |

**Junk Type Description**:
- **Pure Symbols/Separator (纯符号/分割线)**: Lines containing only separators (e.g., "♦ ♦ ♦", "...", "---", etc.)
- **Translator Credits (翻译人员名单)**: Lines containing translator, editor, etc. information
- **Website Watermark (网站水印)**: Website watermark text
- **URL Deletion (URL删除)**: Lines containing URLs
- **Other (其他)**: Other types of junk content

**Data Description**:
- This report performed deep cleaning on the updated corpus (20 Chinese + 200 Western)
- Wider cleaning scope, more detailed detection
- Ensures data quality meets training requirements

---

## 3. Model Training Output Files

### 3.1 Model Files

#### `best_model_epoch_X.pt`

**File Paths**:
- `training_output/models/best_model_epoch_1.pt (not saved)`
- `training_output/models/best_model_epoch_2.pt (not saved)`
- `training_output/models/best_model_epoch_3.pt (not saved)`
- `training_output_stage2_classweights/models/best_model_epoch_1.pt (not saved)`
- `training_output_stage2_classweights/models/best_model_epoch_2.pt (not saved)`
- `training_output_stage2_classweights/models/best_model_epoch_3.pt (not saved)`
- `training_output_stage2_classweights_focal/models/best_model_epoch_3.pt` (**Latest Model**)

**File Purpose**:
- Saves the trained BERT classification model
- Contains model weights, optimizer state, training configuration, etc.
- Used for subsequent text classification prediction

**File Format**: PyTorch model file (.pt)

**File Content**:
- `model_state_dict`: Model weight parameters
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch number
- `best_f1`: Best F1-Score
- `config`: Training configuration parameters

**Recommended Usage**:
- **Best Model**: `training_output_stage2_classweights_focal/models/best_model_epoch_3.pt`
- **Performance Metrics**: Accuracy 99.99%, F1-Score 99.99% (Macro F1 99.99%)


---

### 3.2 Evaluation Report Files

#### `classification_report_*.txt`

**File Paths**:
- `training_output/reports/classification_report_best_model_epoch_3.txt`
- `training_output_stage2_classweights/reports/classification_report_best_model_epoch_3.txt`
- `training_output_stage2_classweights_focal/reports/classification_report_stage2_classweights_focal.txt`

**File Purpose**:
- Detailed classification performance report
- Contains precision, recall, F1-Score, etc. for each class

**File Format**: Plain text (TXT)

**Example File Content**:
```
              Precision    Recall    F1-Score    Support
Western_SciFi        0.9998     0.9999     0.9998     27529
Chinese_Xianxia      0.9999     0.9999     0.9999     59858
Accuracy                                    0.9999     87387
```

**Metric Description**:
- **precision**: The proportion of samples predicted as positive that are truly positive
- **recall**: The proportion of truly positive samples correctly predicted
- **f1-score**: The harmonic mean of precision and recall
- **support**: The number of samples for this class in the validation set

---

#### `metrics_*.json`

**File Paths**:
- `training_output/reports/metrics_best_model_epoch_3.json`
- `training_output_stage2_classweights/reports/metrics_stage2_classweights.json`
- `training_output_stage2_classweights/reports/metrics_best_model_epoch_3.json`
- `training_output_stage2_classweights_focal/reports/metrics_stage2_classweights_focal.json`

**File Purpose**:
- Saves all evaluation metrics in JSON format
- Facilitates programmatic reading and analysis

**File Format**: JSON

**File Content Structure**:
```json
{
  "overall_accuracy": 0.9998970098527241,
  "roc_auc": 0.9999996783650521,
  "pr_auc": 0.9999998524443298,
  "macro_f1": 0.999880682558935,
  "weighted_f1": 0.9998970113679714,
  "class_0": {
    "precision": 1.0,
    "recall": 0.9998910240110429,
    "f1": 0.9999455090364181
  },
  "class_1": {
    "precision": 1.0,
    "recall": 0.9998997627718935,
    "f1": 0.9999498788739454
  },
  "confusion_matrix": [[27526, 3], [6, 59852]],
  "avg_confidence": 0.9989607334136963,
  "val_loss": 0.0004991283183475088,
  "total_training_time": 13424.593098402023
}
```

**Field Description**:
- `overall_accuracy`: Overall accuracy
- `roc_auc`: Area under the ROC curve
- `pr_auc`: Area under the PR curve
- `macro_f1`: Macro-averaged F1-Score
- `weighted_f1`: Weighted average F1-Score
- `class_0`: Metrics for the Western_SciFi class
- `class_1`: Metrics for the Chinese_Xianxia class
- `confusion_matrix`: Confusion matrix
- `avg_confidence`: Average prediction confidence
- `val_loss`: Validation set loss

---

#### `metrics_summary.json`

**File Paths**:
- `training_output/reports/metrics_summary.json`
- `training_output_stage2_classweights/reports/metrics_summary.json`

**File Purpose**:
- Aggregates evaluation metrics from all training epochs
- Facilitates comparison of performance across different epochs

**File Format**: JSON array

**File Content Structure**:
```json
[
  {
    "epoch": 1,
    "train_loss": 0.0021,
    "val_loss": 0.0018,
    "val_f1_macro": 0.9983,
    "val_f1_weighted": 0.9985
  },
  {
    "epoch": 2,
    "train_loss": 0.0003,
    "val_loss": 0.0005,
    "val_f1_macro": 0.9998,
    "val_f1_weighted": 0.9998
  },
  {
    "epoch": 3,
    "train_loss": 0.0001,
    "val_loss": 0.0005,
    "val_f1_macro": 0.9999,
    "val_f1_weighted": 0.9999
  }
]
```


---

#### `confusion_matrix_*.png`

**File Paths**:
- `training_output/reports/confusion_matrix_best_model_epoch_3.png`
- `training_output_stage2_classweights/reports/confusion_matrix_best_model_epoch_3.png`
- `training_output_stage2_classweights_focal/reports/confusion_matrix_stage2_classweights_focal.png`

**File Purpose**:
- Confusion matrix visualization image
- Visually displays the model's prediction accuracy for each class

**File Format**: PNG image file

**Image Content**:
- X-axis: Predicted Label
- Y-axis: True Label
- Matrix values: Number of predictions for each class
- Color intensity: Indicates quantity

**Interpretation**:
- Larger values on the diagonal are better (indicating correct predictions)
- Smaller values off the diagonal are better (indicating incorrect predictions)

---

#### `roc_pr_curves_*.png`

**File Paths**:
- `training_output/reports/roc_pr_curves_best_model_epoch_3.png`
- `training_output_stage2_classweights/reports/roc_pr_curves_best_model_epoch_3.png`
- `training_output_stage2_classweights_focal/reports/roc_pr_curves_stage2_classweights_focal.png`

**File Purpose**:
- Visualization images of ROC curve and PR curve
- Evaluates the model's classification performance

**File Format**: PNG image file

**Image Content**:
- **ROC Curve**: Shows the relationship between True Positive Rate (TPR) and False Positive Rate (FPR)
- **PR Curve**: Shows the relationship between Precision and Recall
- AUC value: Area under the curve, the closer to 1.0 the better

**Interpretation**:
- ROC AUC close to 1.0 indicates excellent model performance
- PR AUC close to 1.0 indicates excellent model performance on positive class prediction

---

#### `training_curves_*.png`

**File Paths**:
- `training_output_stage2_classweights/reports/training_curves_stage2_classweights.png`
- `training_output_stage2_classweights_focal/reports/training_curves_stage2_classweights_focal.png`

**File Purpose**:
- Training process curve chart
- Shows the change of training loss and validation loss across training epochs

**File Format**: PNG image file

**Image Content**:
- Training Loss curve
- Validation Loss curve
- X-axis: Training Epoch
- Y-axis: Loss value

**Interpretation**:
- Loss values should gradually decrease across training epochs
- Training loss and validation loss should remain close to avoid overfitting

---

#### `error_samples_*.csv`

**File Paths**:
- `training_output_stage2_classweights/reports/error_samples_stage2_classweights.csv`
- `training_output_stage2_classweights_focal/reports/error_samples_stage2_classweights_focal.csv` (**Latest**)

**File Purpose**:
- Records samples that the model predicted incorrectly
- Used to analyze the model's error patterns and improvement directions

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `text` | String | Text content | "My case didn't really interest him..." |
| `true_label` | String | True label | "Western_SciFi" |
| `predicted_label` | String | Predicted label | "Chinese_Xianxia" |
| `confidence` | Float | Prediction confidence | 0.9998 |
| `prob_class_0` | Float | Probability for class 0 | 0.0001 |
| `prob_class_1` | Float | Probability for class 1 | 0.9999 |

**Data Description**:
- Contains only incorrectly predicted samples
- Confidence is usually very high (close to 1.0), indicating the model is also "confident" about incorrect predictions
- Can be used to analyze boundary cases and model limitations
- The updated model has 9 error samples total (3 Western_SciFi→Chinese_Xianxia, 6 Chinese_Xianxia→Western_SciFi)


---

## 4. Topic Analysis Data Files

This section contains all data files generated by using BERTopic for unsupervised topic modeling and sentiment analysis.

> **Note**: The following data is based on re-performed unsupervised clustering analysis on the updated corpus (20 Chinese + 200 Western, 436,934 documents in total).

---

### 4.1 `topic_analysis.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis.csv`

**File Purpose**:
- Main result file of topic modeling
- Contains keywords, frequency, distribution, etc. for each topic
- Used to quickly understand the main topics in the text data

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `topic_id` | Integer | Topic ID (-1 indicates noise/unclassified) | 0, 1, 2, ... |
| `topic_keywords` | String | Topic keywords (comma-separated) | "experts, space, skeleton, mental, origin..." |
| `topic_keywords_with_scores` | String | Keywords with c-TF-IDF scores | "experts(0.2247); space(0.2234); skeleton(0.2122)..." |
| `frequency` | Integer | Number of documents in this topic | 26444 |
| `percentage` | Float | Document percentage of this topic (%) | 6.05 |
| `source_distribution` | String | Source distribution (document count per category) | "Chinese_Xianxia: 26235, Western_SciFi: 209" |
| `files` | String | List of files containing this topic | "clean_Swallowed Star.txt, clean_Foundation Trilogy.txt..." |

**Data Description**:
- **topic_id = -1**: Represents the noise topic, containing documents that cannot be classified into any topic
- **topic_id ≥ 0**: Represents valid topics, sorted by frequency from high to low
- **Keyword Scores**: The higher the c-TF-IDF score, the more representative the word is of the topic

**Data Statistics**:
- Total topics: 159 (158 valid topics + 1 noise topic)
- Total documents: 436,934
- Largest topic (topic_id=0): 26,444 documents (6.05%)
- Noise topic (topic_id=-1): 55,689 documents (12.75%)
- Valid topic document count: 381,245 (87.25%)
- Smallest valid topic (topic_id=157): 171 documents (0.04%)


**Notes**:
- Compared to the old version (49 topics, 49.16% noise), the updated version has a significantly increased number of topics (158) and a significantly reduced noise ratio (12.75%), indicating that the larger corpus provides richer topic signals
- Keywords help understand the content characteristics of each topic
- The source_distribution field can be used to analyze the distribution of topics across different categories

---

### 4.2 `topic_analysis_detailed_topics.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_detailed_topics.csv`

**File Purpose**:
- Detailed information about topics, including representative documents
- Used for in-depth understanding of the specific content of each topic
- Contains representative document chunks for each topic

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `Topic` | Integer | Topic ID | 0 |
| `Count` | Integer | Number of documents | 26444 |
| `Name` | String | Topic name (based on keywords) | "0_experts_space_skeleton_mental" |
| `Representation` | String | Topic representation (keyword list) | "['experts', 'space', 'skeleton', 'mental'...]" |
| `Representative_Docs` | String | Representative documents (JSON format) | "['Document 1...', 'Document 2...', 'Document 3...']" |

**Data Description**:
- **Representative_Docs**: Contains the document chunks most representative of the topic (usually 3)
- These documents can help understand the actual content of the topic
- Documents are stored in JSON array format


---

### 4.3 `topic_analysis_document_topics.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_document_topics.csv`

**File Size**: ~620MB (contains the complete text of all documents)

**File Purpose**:
- Topic assignment table for each document
- Records which topic each document is assigned to
- Contains complete document text (for subsequent sentiment analysis)

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `document_id` | Integer | Document ID (starting from 0) | 0, 1, 2, ... |
| `text_preview` | String | Text preview (first 200 characters) | "Before they left, Exquisite told..." |
| `text` | String | Complete text content | "Complete document text..." |
| `topic_id` | Integer | Assigned topic ID | 0, 1, 2, -1 |
| `topic_probability` | Float | Topic assignment probability (optional) | 0.85 |
| `source_file` | String | Source file name | "clean_Swallowed Star.txt" |
| `source` | String | Source category (optional) | "Chinese_Xianxia" |
| `timestamp` | String | Timestamp (book publication/completion/award year) | "2016" |

**Data Description**:
- Each document is assigned to one topic (including noise topic -1)
- `topic_probability` represents the confidence of the document belonging to that topic
- The `text` field contains the complete text; the file is large


**Notes**:
- The file is very large (>200MB), containing the complete text of all documents
- It is recommended to use the `nrows` parameter to limit rows or use chunked reading
- This is the input file for sentiment analysis

---

### 4.4 `topic_sentiment_final.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_sentiment_final.csv`

**File Purpose**:
- Sentiment analysis statistical results for each topic
- Comprehensive result combining topic modeling and sentiment analysis
- Used to analyze the sentiment tendencies of different topics

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `Topic` | Integer | Topic ID | 0 |
| `Sentiment_Mean` | Float | Mean sentiment score (-1 to 1) | -0.1983 |
| `Sentiment_Std` | Float | Standard deviation of sentiment scores | 0.3378 |
| `Frequency` | Integer | Number of documents in this topic | 26444 |
| `Positive_Count` | Integer | Number of positive sentiment documents | 6660 |
| `Negative_Count` | Integer | Number of negative sentiment documents | 19784 |
| `Neutral_Count` | Integer | Number of neutral sentiment documents | 0 |
| `Keywords` | String | Topic keyword identifier | "Topic 0" |

**Sentiment Score Description**:
- **Sentiment_Mean > 0**: Overall sentiment tendency is positive
- **Sentiment_Mean < 0**: Overall sentiment tendency is negative
- **Sentiment_Mean ≈ 0**: Overall sentiment tendency is neutral
- **Sentiment_Std**: Larger standard deviation indicates more dispersed sentiment distribution within the topic

**Sentiment Classification Criteria**:
- **Positive_Count**: Number of documents with sentiment score > 0
- **Negative_Count**: Number of documents with sentiment score < 0
- **Neutral_Count**: Number of documents with sentiment score = 0 (usually 0)

**Data Interpretation Example**:
- Topic 0: Mean sentiment -0.198 (slightly negative), 6,660 positive documents, 19,784 negative documents
- Indicates this topic leans negative overall, but with large internal variation


---

### 4.5 `temp_sentiment_docs.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/temp_sentiment_docs.csv`

**File Size**: >200MB (contains sentiment scores for all documents)

**File Purpose**:
- Intermediate result file for document-level sentiment scores
- Records the sentiment score (-1 to 1) for each document
- Used to generate topic-level sentiment statistics

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `document_id` | Integer | Document ID | 0, 1, 2, ... |
| `text` | String | Document text | "Complete document text..." |
| `sentiment_score` | Float | Sentiment score (-1 to 1) | -0.2345 |
| `topic_id` | Integer | Topic ID | 0 |
| `source_file` | String | Source file name | "clean_Swallowed Star.txt" |

**Sentiment Score Description**:
- **-1.0**: Extreme negative
- **0.0**: Neutral
- **1.0**: Extreme positive
- Calculated using the RoBERTa-base-sentiment model

**Notes**:
- This is an intermediate result file, usually not saved (`SAVE_TEMP_FILE = False`)
- **This file does not currently exist in the project** because the default configuration does not enable saving
- If this file is needed, set `SAVE_TEMP_FILE = True` when running sentiment analysis
- The file is large, containing the complete text of all documents


---

### 4.6 Topic Visualization Files

#### `topic_barchart.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/topic_barchart.html`

**File Purpose**:
- Topic frequency bar chart (interactive)
- Visualizes the document count and keywords for each topic

**File Format**: HTML (generated using Plotly)

**Visualization Content**:
- X-axis: Topic ID
- Y-axis: Document count
- Each bar shows the topic's keywords
- Supports interactive operations (zoom, hover for details)

**Usage**:
- Open the HTML file directly in a browser
- Hover to see detailed information
- Can be zoomed and filtered

---

#### `topic_distance.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/topic_distance.html`

**File Purpose**:
- Topic distance visualization (2D mapping after dimensionality reduction)
- Shows similarity and relationships between topics

**File Format**: HTML (interactive)

**Visualization Content**:
- Each circle represents a topic; circle size represents topic frequency
- Closer topics are more similar
- Interactive chart, supports zoom and hover for details
- Click to view topic details

**Interpretation**:
- Topics clustered together may be related
- Isolated topics may be unique
- Noise topic (-1) is usually far from other topics

---

#### `topic_hierarchy.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/topic_hierarchy.html`

**File Purpose**:
- Topic hierarchy visualization
- Shows hierarchical relationships between topics

**File Format**: HTML (interactive)

**Visualization Content**:
- Tree structure displays topic hierarchy
- Can be expanded/collapsed to view different levels
- Helps understand the organizational structure of topics

---

#### `ctfidf_scores.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/ctfidf_scores.html`

**File Purpose**:
- c-TF-IDF score visualization
- Shows the importance of keywords in each topic

**File Format**: HTML (interactive)

**Visualization Content**:
- Keywords and their c-TF-IDF scores for each topic
- Higher scores indicate stronger representation of the topic
- Can be filtered by topic

---

#### `topic_similarity_matrix.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/topic_similarity_matrix.html`

**File Purpose**:
- Topic similarity matrix visualization
- Shows the degree of similarity between different topics

**File Format**: HTML (interactive Plotly chart)

**Visualization Content**:
- Heatmap form displays cosine similarity between topics
- Darker colors indicate more similar topics
- Can be used to discover similar topics that can be merged

---

#### `topic_comparison_bar.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/topic_comparison_bar.html`

**File Purpose**:
- Comparison of topic distribution across different sources (Chinese/Western)
- Visually shows the source composition of each topic

**File Format**: HTML (interactive Plotly chart)

**Visualization Content**:
- Stacked bar chart shows the source distribution of topics
- Distinguishes the contributions of Chinese_Xianxia and Western_SciFi
- Can be used to analyze which topics are unique to a particular category

---

#### `topic_comparison_data.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/topic_comparison_data.csv`

**File Purpose**:
- Raw data file for topic source comparison
- Contains the number of documents per topic in different sources (Chinese/Western)
- Is the data source for the `topic_comparison_bar.html` visualization chart

**File Format**: CSV

---

#### `documents_and_topics.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_output/documents_and_topics.html`

**File Purpose**:
- Interactive visualization of document-topic distribution
- Shows the distribution of documents in topic space

**File Format**: HTML (interactive Plotly chart)

---

#### `topics_over_time.html` / `topics_over_time.csv`

**File Paths**:
- `bertopic_analysis/topic_analysis_output/topics_over_time.html`
- `bertopic_analysis/topic_analysis_output/topics_over_time.csv`

**File Purpose**:
- Shows the trend of topics over time (work publication/completion/award time)
- Analyzes the popularity evolution of topics in different periods

**File Format**: HTML (interactive chart) + CSV (raw data)

---

#### `sentiment_quadrant_chart.html`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/sentiment_quadrant_chart.html`

**File Purpose**:
- Sentiment-Topic quadrant chart
- Simultaneously displays the frequency and sentiment tendency of topics

**File Format**: HTML (interactive Plotly chart)

**Visualization Content**:
- **X-axis**: Topic frequency (document count)
- **Y-axis**: Mean sentiment score (-1 to 1)
- **Four Quadrants**:
  - Top-right: High frequency + positive sentiment
  - Bottom-right: High frequency + negative sentiment
  - Top-left: Low frequency + positive sentiment
  - Bottom-left: Low frequency + negative sentiment
- Each point represents a topic; size indicates document count

**Interpretation**:
- Top-right quadrant: Important and positive topics
- Bottom-right quadrant: Important but negative topics
- Click to view topic details

**Usage**:
- Open directly in a browser
- Hover to see detailed topic information
- Can be zoomed and filtered

---

## 5. Original Text Files

### 5.1 Cleaned Text Files

Cleaned text files are pure novel text that has undergone data cleaning, with all junk content removed (such as translator credits, website watermarks, URL links, pure separators, etc.). These files are the direct data source for the training dataset and topic analysis.

---

#### 5.1.1 `Cleaned_Chinese_Trans_V3/` Directory

**File Path**: `/root/autodl-tmp/clean/Cleaned_Chinese_Trans_V3/`

**Directory Purpose**:
- Stores intermediate transition files for Chinese web novels (Chinese_Xianxia)
- **Note**: These files are only intermediate products and are **not directly used by `3.py`** (`3.py` uses the deeply cleaned `Deep_Cleaned_Chinese`)
- Generation method: First manual cleaning (preserve main text, delete unrelated content such as acknowledgments), then run the `2.py` script to generate

**File List** (partial example):
1. `clean_A Will Eternal.txt` - 《一念永恒》 (translated version)
2. `clean_Battle Through the Heavens.txt` - 《斗破苍穹》 (translated version)
3. `clean_Coiling Dragon.txt` - 《盘龙》 (translated version)
4. `clean_Cultivation Chat Group.txt` - 《修真聊天群》 (translated version)
5. `clean_Desolate Era.txt` - 《莽荒纪》 (translated version)
6. `clean_Forty Millenniums of Cultivation.txt` - 《修真四万年》 (translated version)
7. `clean_I Shall Seal the Heavens.txt` - 《我欲封天》 (translated version)
8. `clean_Legendary Mechanic.txt` - 《传奇机械师》 (translated version)
9. `clean_Library of Heaven is Path.txt` - 《天道图书馆》 (translated version)
10. `clean_Lord of the Mysteries.txt` - 《诡秘之主》 (translated version)
...(20 books in total)

**File Format**:
- Plain text file (.txt)
- UTF-8 encoding
- Paragraphs separated by two newline characters (`\n\n`)

**Data Features**:
- All translator credits removed (e.g., "Translator: xxx")
- All website watermarks removed (e.g., "NovelFull", "read on webnovel", etc.)
- URL links and donation request information removed
- Pure separators removed (e.g., "***", "---", etc.)
- Repeated chapter titles removed
- Complete main text content retained

**File Size**:
- Single file size: ranging from a few MB to tens of MB
- Total size: approximately 500-800MB (due to increased book count)


**Important Notes**:
- **Manual Cleaning Workflow**: Must **perform manual cleaning first**, then run the `2.py` script.
- **Prohibited Operation**: Do not run the script first and then manually clean the output files.
- **Overwrite Risk**: If you accidentally execute `python 2.py` again after manual cleaning, the contents of the output folder (`Cleaned_Chinese_Trans_V3`) will be overwritten, and **the hard-earned manually cleaned content will be lost** (if manual cleaning was done in the output folder).
- Recommendation: Manual cleaning should be done on the source files, or back up the manually cleaned files before running the script.

---

#### 5.1.2 `Cleaned_English_V2/` Directory

**File Path**: `/root/autodl-tmp/clean/Cleaned_English_V2/`

**Directory Purpose**:
- Stores intermediate transition files for Western sci-fi (Western_SciFi)
- **Note**: These files are only intermediate products and are **not directly used by `3.py`** (`3.py` uses the deeply cleaned `Deep_Cleaned_English`)
- Generation method: First manual cleaning (preserve main text, delete unrelated content such as acknowledgments), then run the `1.py` script to generate

**File List** (partial example):
1. `clean_253.txt` - 《253》 (Geoff Ryman)
2. `clean_A Clash of Kings.txt` - 《列王的纷争》 (George R.R. Martin)
3. `clean_Anansi Boys.txt` - 《阿南西之子》 (Neil Gaiman)
4. `clean_Babel.txt` - 《通天塔》 (R.F. Kuang)
5. `clean_Dune.txt` - 《沙丘》 (Frank Herbert)
6. `clean_Foundation Trilogy.txt` - 《基地三部曲》 (Isaac Asimov)
7. `clean_Game of Thrones.txt` - 《权力的游戏》 (George R.R. Martin)
8. `clean_Harry Potter.txt` - 《哈利波特》 (J.K. Rowling)
9. `clean_Neuromancer.txt` - 《神经漫游者》 (William Gibson)
10. `clean_The Fifth Season.txt` - 《第五季》 (N.K. Jemisin)
...(200 books in total, covering various genres including classic sci-fi, fantasy, cyberpunk, etc.)

**File Format**:
- Plain text file (.txt)
- UTF-8 encoding
- Paragraphs separated by two newline characters (`\n\n`)

**Data Features**:
- All website watermarks removed (e.g., "NovelFull", "www.feedbooks.com", etc.)
- URL links removed
- Table of contents navigation content removed
- Highly repeated long sentence content removed
- Copyright notices (partially) removed
- Complete main text content retained

**File Size**:
- Single file size: ranging from a few MB to tens of MB
- Total size: approximately 2-4GB (due to significantly increased book count)


**Important Notes**:
- **Manual Cleaning Workflow**: Must **perform manual cleaning first**, then run the `1.py` script.
- **Prohibited Operation**: Do not run the script first and then manually clean the output files.
- **Overwrite Risk**: If you accidentally execute `python 1.py` again after manual cleaning, the contents of the output folder (`Cleaned_English_V2`) will be overwritten, and **the hard-earned manually cleaned content will be lost** (if manual cleaning was done in the output folder).
- Recommendation: Manual cleaning should be done on the source files, or back up the manually cleaned files before running the script.

---

### 5.2 Original Uncleaned Text Files (Optional Reference)

**File Paths**:
- `/root/autodl-tmp/clean/英文中方/` - Original Chinese web novel files (uncleaned)
- `/root/autodl-tmp/clean/英文西方/` - Original Western sci-fi files (uncleaned)

**Directory Purpose**:
- Stores originally downloaded text files (without cleaning processing)
- Serves as a backup before data cleaning
- Used to compare differences before and after cleaning

**File Features**:
- Contains all the original text content
- May contain junk content such as translator credits, website watermarks, URL links
- Format may not be uniform
- Encoding may be inconsistent (UTF-8 or Latin-1)

**Notes**:
- These files are only for backup and reference
- Actual training and analysis use the deeply cleaned files (`Deep_Cleaned_Chinese/` and `Deep_Cleaned_English/`)
- It is not recommended to use these files directly for training

---

### 5.3 Deeply Cleaned Text Files

#### 5.3.1 `Deep_Cleaned_Chinese/` Directory

**File Path**: `/root/autodl-tmp/clean/Deep_Cleaned_Chinese/`

**Directory Purpose**:
- Stores Chinese web novel text files after deep cleaning processing
- Based on `Cleaned_Chinese_Trans_V3/`, with more detailed cleaning performed
- Contains 20 Chinese web novels

**File Format**:
- Plain text file (.txt)
- UTF-8 encoding
- File naming consistent with `Cleaned_Chinese_Trans_V3/`

**Data Features**:
- Deep cleaning performed based on issues recorded in `deep_clean_report.csv`
- Further removed residual symbol separators, watermarks, etc.
- File count: 20

---

#### 5.3.2 `Deep_Cleaned_English/` Directory

**File Path**: `/root/autodl-tmp/clean/Deep_Cleaned_English/`

**Directory Purpose**:
- Stores Western sci-fi text files after deep cleaning processing
- Based on `Cleaned_English_V2/`, with more detailed cleaning performed
- Contains 200 Western sci-fi novels

**File Format**:
- Plain text file (.txt)
- UTF-8 encoding
- File naming consistent with `Cleaned_English_V2/`

**Data Features**:
- Deep cleaning performed based on issues recorded in `deep_clean_report.csv`
- Further removed residual symbol separators, watermarks, etc.
- File count: 200

**Notes**:
- The `Deep_Cleaned_*` directories are the final cleaned versions used
- The training dataset `bert_training_dataset.csv` is generated based on these deeply cleaned files

---

### 5.4 Timestamp Reference Files

#### `获奖(完结)时间/` (Award/Completion Time) Directory

**File Path**: `/root/autodl-tmp/clean/获奖(完结)时间/`

**Directory Purpose**:
- Stores reference materials for various award times and Chinese novel completion times
- Used to provide timestamp information for the `Timestamps` field in the training data

**File List**:
1. `Hugo奖获奖时间.md` - Hugo Award winning works time records
2. `Locus奖获奖时间.md` - Locus Award winning works time records
3. `Nebula奖获奖时间.md` - Nebula Award winning works time records
4. `菲利普迪克奖获奖时间.md` - Philip K. Dick Award winning works time records
5. `中文小说完结时间.md` - Chinese web novel completion time records

**File Format**: Markdown (.md)

**Data Description**:
- Timestamps for Western sci-fi works are mainly based on award times (Hugo, Nebula, Locus, Philip K. Dick Award)
- Timestamps for Chinese web novel works are based on novel completion times
- This time data is used by the `add_timestamps.py` script to add the `Timestamps` field to the training data

---

### 5.5 Project Scripts and Documentation Files

The following are Python scripts and documentation files in the project root directory, used for data processing, model training, and analysis.

**Python Scripts**:

| File Name | Purpose | Description |
|-----------|---------|-------------|
| `1.py` | Data processing script | Related to data preprocessing |
| `2.py` | Data processing script | Related to data cleaning |
| `3.py` | Data processing script | Data chunking/training set generation |
| `4.py` | Helper script | Auxiliary data processing |
| `add_timestamps.py` | Timestamp addition | Adds the `Timestamps` field to training data based on reference materials in `获奖(完结)时间/` |
| `check_clean.py` | Cleaning quality check | Checks whether junk content still exists in cleaned files; generates `clean_check_report.csv` |
| `check_gpu.py` | GPU check | Checks GPU availability and CUDA environment |
| `download_model_local.py` | Model download | Downloads pre-trained models to a local directory |
| `download_roberta_sentiment_model.py` | Sentiment model download | Downloads the RoBERTa sentiment analysis model locally |
| `evaluate_saved_models.py` | Model evaluation | Evaluates saved trained models |
| `predict.py` | Prediction script | Performs text classification prediction using trained models |
| `replot_training_curves.py` | Replot training curves | Re-plots the loss curves of the training process |
| `train_bert.py` | BERT training | Main training script for the BERT classification model |

**BERTopic Analysis Scripts** (located in the `bertopic_analysis/` directory):

| File Name | Purpose | Description |
|-----------|---------|-------------|
| `topic_modeling.py` | Topic modeling | Main script for BERTopic topic modeling |
| `sentiment_analysis.py` | Sentiment analysis | Computes sentiment scores for topic documents |
| `visualize_topic_comparison.py` | Visualization comparison | Generates topic source comparison visualization charts |
| `stopwords.py` | Stopwords definition | Defines the stopword list used by BERTopic |

**Documentation Files**:

| File Name | Purpose | Description |
|-----------|---------|-------------|
| `Data_Description_Document.md` | Data file description | This document, describing the purpose and format of all data files |
| `Detailed_Data_Results_Report.md` | Results report | Detailed data analysis and model training results report |
| `Project_Delivery_Document.md` | Project delivery | Overall project delivery documentation |
| `Stopwords_Description.md` | Stopwords description | Documentation for the stopword list |
| `Final_Erroneous_Stopwords_List.md` | Erroneous stopwords | Records the list of words incorrectly labeled as stopwords |
| `requirements.txt` | Dependency list | Project Python dependency package list |
| `bertopic_analysis/requirements_bertopic.txt` | BERTopic dependencies | Dependency package list for the BERTopic analysis module |

---

## 6. Model Configuration Files

### 6.1 Pre-trained Model Files

This project uses three pre-trained deep learning models, which are stored in local directories to avoid downloading from the network during training and inference.

---

#### 6.1.1 BERT Classification Model: `bert-base-uncased-local/`

**File Path**: `/root/autodl-tmp/clean/bert-base-uncased-local/`

**Model Purpose**:
- Used for text classification tasks (distinguishing Western_SciFi from Chinese_Xianxia)
- Serves as the base pre-trained model for the BERT classification model
- Fine-tuned during training

**Model Information**:
- **Model Name**: BERT-base-uncased
- **Model Type**: Bidirectional Encoder
- **Parameter Count**: ~110M
- **Vocabulary Size**: 30,522
- **Max Sequence Length**: 512 tokens
- **Language**: English (case-insensitive)

**Directory Structure**:
```
bert-base-uncased-local/
├── config.json              # Model configuration file
├── model.safetensors        # Model weight file (SafeTensors format)
├── tokenizer_config.json    # Tokenizer configuration
├── vocab.txt                # Vocabulary file
└── special_tokens_map.json  # Special token mapping
```

**File Description**:

| File Name | Description | Purpose |
|-----------|-------------|---------|
| `config.json` | Model architecture configuration | Defines model layers, hidden layer size, number of attention heads, etc. |
| `model.safetensors` | Model weights | Contains all trainable parameters (uses the SafeTensors format, which is safer) |
| `tokenizer_config.json` | Tokenizer configuration | Defines how to convert text into tokens |
| `vocab.txt` | Vocabulary | Contains all possible tokens and their ID mappings |
| `special_tokens_map.json` | Special tokens | Defines special tokens such as [CLS], [SEP], [PAD] |

**Notes**:
- The model files are large (about 400-500MB)
- Using a local model avoids network downloads and increases training speed
- The model has been pre-trained on English text

---

#### 6.1.2 RoBERTa Sentiment Analysis Model: `roberta-base-sentiment-local/`

**File Path**: `/root/autodl-tmp/clean/roberta-base-sentiment-local/`

**Model Purpose**:
- Used for sentiment analysis tasks
- Calculates the sentiment score of text (between -1 and 1)
- Used in topic analysis to analyze the sentiment tendency of each topic

**Model Information**:
- **Model Name**: RoBERTa-base-sentiment
- **Model Type**: RoBERTa (Robustly Optimized BERT)
- **Parameter Count**: ~125M
- **Vocabulary Size**: 50,265
- **Max Sequence Length**: 512 tokens
- **Task**: Sentiment Analysis

**Directory Structure**:
```
roberta-base-sentiment-local/
├── config.json              # Model configuration file
├── model.safetensors        # Model weight file
├── tokenizer_config.json    # Tokenizer configuration
├── tokenizer.json           # Tokenizer file (JSON format)
├── vocab.json               # Vocabulary (JSON format)
├── merges.txt               # BPE merge rules
└── special_tokens_map.json   # Special token mapping
```

**File Description**:

| File Name | Description | Purpose |
|-----------|-------------|---------|
| `config.json` | Model architecture configuration | Defines RoBERTa model architecture parameters |
| `model.safetensors` | Model weights | Contains all trainable parameters |
| `tokenizer.json` | Tokenizer file | Complete tokenizer configuration (JSON format) |
| `vocab.json` | Vocabulary | BPE (Byte Pair Encoding) vocabulary |
| `merges.txt` | BPE merge rules | Defines how to merge characters into tokens |
| `special_tokens_map.json` | Special token mapping | Defines mappings for special tokens |

**Sentiment Score Description**:
- **-1.0**: Extreme negative sentiment
- **0.0**: Neutral sentiment
- **1.0**: Extreme positive sentiment
- The model outputs continuous values that accurately reflect sentiment intensity


**Notes**:
- The model files are large (~500MB)
- Specifically used for sentiment analysis tasks
- Used in the topic analysis module (`sentiment_analysis.py`)

---

#### 6.1.3 Sentence Transformers Model: `all-MiniLM-L6-v2-local/`

**File Path**: `/root/autodl-tmp/clean/all-MiniLM-L6-v2-local/`

**Model Purpose**:
- Used to generate vector representations (embeddings) of text
- Used for document embedding in BERTopic topic modeling
- Converts text into fixed-dimensional vectors (384 dimensions)

**Model Information**:
- **Model Name**: all-MiniLM-L6-v2
- **Model Type**: Sentence Transformers (sentence embedding model)
- **Parameter Count**: ~22M (lightweight)
- **Output Dimension**: 384
- **Max Sequence Length**: 256 tokens
- **Features**: Lightweight, fast, suitable for large-scale text processing

**Directory Structure**:
```
all-MiniLM-L6-v2-local/
├── config.json                      # Model configuration
├── model.safetensors                # Model weights
├── config_sentence_transformers.json  # Sentence Transformers configuration
├── sentence_bert_config.json        # Sentence-BERT configuration
├── modules.json                     # Module configuration
├── tokenizer_config.json            # Tokenizer configuration
├── tokenizer.json                   # Tokenizer file
├── vocab.txt                        # Vocabulary
├── special_tokens_map.json           # Special token mapping
├── README.md                        # Model documentation
├── 1_Pooling/                       # Pooling layer configuration
│   └── config.json
└── 2_Normalize/                     # Normalization layer configuration
    └── config.json
```

**File Description**:

| File Name | Description | Purpose |
|-----------|-------------|---------|
| `config.json` | Base model configuration | Defines the underlying Transformer architecture |
| `model.safetensors` | Model weights | Contains all trainable parameters |
| `config_sentence_transformers.json` | ST configuration | Configuration for the Sentence Transformers framework |
| `sentence_bert_config.json` | SBERT configuration | Sentence-BERT specific configuration |
| `modules.json` | Module configuration | Defines the modular structure of the model |
| `tokenizer_config.json` | Tokenizer configuration | Defines how to convert text into tokens |
| `tokenizer.json` | Tokenizer file | Complete tokenizer configuration (JSON format) |
| `vocab.txt` | Vocabulary | Contains all possible tokens and their ID mappings |
| `special_tokens_map.json` | Special tokens | Defines mappings for special tokens |
| `README.md` | Model documentation | Detailed model documentation |
| `1_Pooling/config.json` | Pooling layer configuration | Defines how to generate sentence embeddings from token embeddings |
| `2_Normalize/config.json` | Normalization layer configuration | Defines how to normalize output vectors |


**Notes**:
- The model files are small (~80-90MB), suitable for fast processing
- Specifically optimized for generating sentence-level embeddings
- Used in BERTopic topic modeling (`topic_modeling.py`)

---

### 6.2 Training Configuration Files

Training configuration files record the detailed parameter settings of each model training, making it easy to reproduce training results and compare the effects of different training configurations.

---

#### 6.2.1 `training_config_stage1.json`

**File Path**:
- `training_output/reports/training_config_stage1.json`

**File Purpose**:
- Records the configuration parameters of the first training stage
- Used to reproduce training results
- Facilitates comparison of configuration differences between different training stages

**File Format**: JSON

**Example Configuration Content**:
```json
{
  "model_name": "/root/autodl-tmp/clean/bert-base-uncased-local",
  "batch_size": 16,
  "learning_rate": 2e-05,
  "epochs": 3,
  "max_length": 512,
  "mode": "stage1",
  "downsample": false,
  "use_class_weights": false,
  "use_focal_loss": false,
  "train_samples": 349547,
  "val_samples": 87387,
  "class_names": ["Western_SciFi", "Chinese_Xianxia"],
  "label_map": {
    "Western_SciFi": 0,
    "Chinese_Xianxia": 1
  },
  "timestamp": "2026-02-12 09:00:00"
}
```

**Field Description**:

| Field Name | Description | Example Value |
|------------|-------------|---------------|
| `model_name` | Pre-trained model path | `bert-base-uncased-local` |
| `batch_size` | Batch size | `16` |
| `learning_rate` | Learning rate | `2e-05` (0.00002) |
| `epochs` | Number of training epochs | `3` |
| `max_length` | Maximum sequence length | `512` tokens |
| `mode` | Training mode | `stage1` (first stage) |
| `downsample` | Whether to downsample | `false` (no downsampling) |
| `use_class_weights` | Whether to use class weights | `false` (not used) |
| `use_focal_loss` | Whether to use Focal Loss | `false` (not used) |
| `train_samples` | Number of training samples | `349547` |
| `val_samples` | Number of validation samples | `87387` |
| `class_names` | List of class names | `["Western_SciFi", "Chinese_Xianxia"]` |
| `label_map` | Label mapping | `{"Western_SciFi": 0, "Chinese_Xianxia": 1}` |
| `timestamp` | Training timestamp | `"2026-02-12 09:00:00"` |

**Training Features**:
- First stage: Basic training without class weights
- The class imbalance problem may cause the model to bias toward the majority class (Chinese_Xianxia)

---

#### 6.2.2 `training_config_stage2_classweights.json`

**File Path**:
- `training_output_stage2_classweights/reports/training_config_stage2_classweights.json`

**File Purpose**:
- Records the configuration parameters of the second training stage (using class weights)
- This is the training configuration for the **second stage model** (without Focal Loss)
- Used to reproduce the training results of this stage

**File Format**: JSON

**Example Configuration Content**:
```json
{
  "model_name": "/root/autodl-tmp/clean/bert-base-uncased-local",
  "batch_size": 16,
  "learning_rate": 2e-05,
  "epochs": 3,
  "max_length": 512,
  "mode": "stage2",
  "downsample": false,
  "use_class_weights": true,
  "use_focal_loss": false,
  "train_samples": 349547,
  "val_samples": 87387,
  "class_names": ["Western_SciFi", "Chinese_Xianxia"],
  "label_map": {
    "Western_SciFi": 0,
    "Chinese_Xianxia": 1
  },
  "timestamp": "2026-02-12 09:30:00"
}
```

**Field Description**:
- Basically the same as stage1, with main differences:
  - `mode`: `"stage2"` (second stage)
  - `use_class_weights`: `true` (**use class weights**)

**Training Features**:
- Second stage: Uses class weights to balance the class imbalance problem
- Class weight calculation: Weights are automatically calculated based on class frequency
  - Western_SciFi (minority class): higher weight
  - Chinese_Xianxia (majority class): lower weight
- **This is one of the high-performance model configurations**, with an accuracy of 99.99% and F1-Score of 99.99% (comparable to stage2_focal)

---

#### 6.2.3 `training_config_stage2_classweights_focal.json`

**File Path**:
- `training_output_stage2_classweights_focal/reports/training_config_stage2_classweights_focal.json`

**File Purpose**:
- Records the configuration parameters of the third training stage (using class weights + Focal Loss)
- Experimental training configuration for comparing the effects of different loss functions
- Used to reproduce the results of this training stage

**File Format**: JSON

**Example Configuration Content**:
```json
{
  "model_name": "bert-base-uncased-local",
  "batch_size": 16,
  "learning_rate": 2e-05,
  "epochs": 3,
  "max_length": 512,
  "mode": "stage2",
  "downsample": false,
  "use_class_weights": true,
  "use_focal_loss": true,
  "train_samples": 349547,
  "val_samples": 87387,
  "class_names": ["Western_SciFi", "Chinese_Xianxia"],
  "label_map": {
    "Western_SciFi": 0,
    "Chinese_Xianxia": 1
  },
  "timestamp": "2026-02-12 09:31:48"
}
```

**Field Description**:
- Basically the same as stage2_classweights, with the main difference:
  - `use_focal_loss`: `true` (**use Focal Loss**)

**Training Features**:
- Third stage: Uses both class weights and Focal Loss simultaneously
- Focal Loss: A loss function specifically designed to handle class imbalance
- Performance metrics: Accuracy 99.99%, F1-Score 99.99% (Macro F1 99.99%)
- Similar performance to stage2_classweights, but training time may differ slightly

**Focal Loss Description**:
- Focal Loss makes the model focus more on hard-to-classify samples by reducing the weight of easy-to-classify samples
- The gamma parameter controls the weight difference between hard and easy samples
- Usually used in combination with class weights for better results

---

### 6.3 Other Training Output Files

#### 6.3.1 `error_samples_stage2_classweights_focal.csv`

**File Path**:
- `training_output_stage2_classweights_focal/reports/error_samples_stage2_classweights_focal.csv`

**File Purpose**:
- Records samples that the model trained with Focal Loss predicted incorrectly
- Used to analyze the impact of Focal Loss on error patterns
- Compare error sample differences between different loss functions

**File Format**: CSV

**Field Description**:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `text` | String | Text content | "I giggled hysterically..." |
| `true_label` | String | True label | "Chinese_Xianxia" |
| `predicted_label` | String | Predicted label | "Western_SciFi" |
| `confidence` | Float | Prediction confidence | 0.7571 |
| `prob_class_0` | Float | Probability for class 0 | 0.7571 |
| `prob_class_1` | Float | Probability for class 1 | 0.2429 |

**Data Description**:
- Contains only incorrectly predicted samples
- Can be compared with `error_samples_stage2_classweights.csv` to analyze the impact of Focal Loss
- Number of error samples is typically very small (<10)

---

#### 6.3.2 `metrics_stage2_classweights_focal.json`

**File Path**:
- `training_output_stage2_classweights_focal/reports/metrics_stage2_classweights_focal.json`

**File Purpose**:
- Records detailed evaluation metrics for the model trained with Focal Loss
- Contains performance metrics for each training epoch
- Used to compare the effects of different loss functions

**File Format**: JSON

**File Content Structure**:
```json
{
  "overall_accuracy": 0.9998970098527241,
  "roc_auc": 0.9999996783650521,
  "pr_auc": 0.9999998524443298,
  "macro_f1": 0.999880682558935,
  "weighted_f1": 0.9998970113679714,
  "class_0": {
    "precision": 1.0,
    "recall": 0.9998910240110429,
    "f1": 0.9999455090364181
  },
  "class_1": {
    "precision": 1.0,
    "recall": 0.9998997627718935,
    "f1": 0.9999498788739454
  },
  "confusion_matrix": [[27526, 3], [6, 59852]],
  "avg_confidence": 0.9989607334136963,
  "total_training_time": 13424.593098402023,
  "epoch_metrics": [...]
}
```

**Field Description**:
- Same structure as `metrics_stage2_classweights.json`
- `total_training_time`: Total training time (seconds)
- `epoch_metrics`: Array of detailed metrics for each training epoch

**Performance Comparison**:
- **stage2_classweights**: Accuracy 99.99%, F1-Score 99.99%
- **stage2_classweights_focal**: Accuracy 99.99%, F1-Score 99.99% (Macro F1 99.99%)
- The two have very similar performance; Focal Loss does not significantly improve performance on this dataset

---

## 7. Data File Usage Guide

This chapter provides detailed field descriptions and usage guidelines for all data files in the project, helping users quickly understand the structure and purpose of each file.

---

### 7.1 Training Data File Field Description

#### 7.1.1 `bert_training_dataset.csv`

**File Path**: `/root/autodl-tmp/clean/bert_training_dataset.csv`

**File Purpose**: The main dataset for BERT model training, containing all preprocessed text chunks and their labels.

**File Format**: CSV (Comma-Separated Values)

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `text` | String | Yes | Text content (cleaned text chunk) | "The starship drifted through the void..." |
| `source` | String | Yes | Text source category | "Western_SciFi" or "Chinese_Xianxia" |
| `book_name` | String | Yes | Source book name | "Dune" or "Lord of the Mysteries" |
| `chunk_id` | Integer | Yes | Chunk number within the book (starting from 1) | 1, 2, 3, ... |
| `char_count` | Integer | Yes | Character count of text (including spaces and punctuation) | 800-1500 (typical) |
| `estimated_tokens` | Integer | Yes | Estimated token count (for BERT, approx. 4 chars = 1 token) | 200-400 (typical) |
| `Timestamps` | String | Yes | Timestamp corresponding to the text (book publication/completion/award year) | "2016" |

**Data Statistics**:
- Total records: 436,934
- Western_SciFi (Western Sci-Fi): 137,645 records (31.5%)
- Chinese_Xianxia (Chinese Web Novels): 299,289 records (68.5%)
- Average character count: ~1,147 characters
- Average token count: ~287 tokens

---

### 7.2 Data Cleaning Report File Field Description

#### 7.2.1 `clean_check_report.csv`

**File Path**: `/root/autodl-tmp/clean/clean_check_report.csv`

**File Purpose**: Records all junk content detected during data cleaning, used for quality control and validation of cleaning effectiveness.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `文件夹` (Folder) | String | Yes | Folder path where the file is located | "Cleaned_Chinese_Trans_V3" or "Cleaned_English_V2" |
| `文件名` (File Name) | String | Yes | Source file name (including extension) | "clean_My House of Horrors.txt" |
| `行号` (Line Number) | Integer | Yes | Line number where junk content is located (starting from 1) | 1, 2, 3, ... |
| `内容` (Content) | String | Yes | Original content marked as junk | "..." or "Translator: xxx" |
| `垃圾类型` (Junk Type) | String | Yes | Classification label for junk content | "纯分割线", "翻译人员名单", "NovelFull水印", "URL删除", "其他" |

**Junk Type Detailed Description**:
- **Pure Separator (纯分割线)**: Lines containing only separators (e.g., "...", "---", "***", etc.)
- **Translator Credits (翻译人员名单)**: Lines containing translator, editor information (e.g., "Translator: StarveCleric")
- **NovelFull Watermark (NovelFull水印)**: Website watermark text (e.g., "read on webnovel", "NovelFull", etc.)
- **URL Deletion (URL删除)**: Lines containing URLs (e.g., "www.feedbooks.com")
- **Other (其他)**: Other types of junk content

---

#### 7.2.2 `deleted_report_cn_v3.csv`

**File Path**: `/root/autodl-tmp/clean/deleted_report_cn_v3.csv`

**File Purpose**: Records content deleted during cleaning of Chinese web novels (Chinese_Xianxia), with detailed records of deletion reasons and locations.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `文件名` (File Name) | String | Yes | Source file name (without path) | "Library of Heaven is Path.txt" |
| `行号` (Line Number) | Integer | Yes | Line number where deleted content is located (starting from 1) | 1, 2, 3, ... |
| `原内容` (Original Content) | String | Yes | Original deleted content (complete line) | "Translator: StarveCleric Editor: Thaddpo" |
| `原因` (Reason) | String | Yes | Deletion reason classification | "翻译人员名单", "NovelFull水印", "URL删除", "纯分割线", "其他" |

**Deletion Reason Type Description**:
- **Translator Credits (翻译人员名单)**: Translator, editor information (e.g., "Translator: xxx", "Editor: xxx")
- **NovelFull Watermark (NovelFull水印)**: Website watermark (e.g., "read on webnovel", "NovelFull", etc.)
- **URL Deletion (URL删除)**: Content containing URLs (e.g., "www.webnovel.com")
- **Pure Separator (纯分割线)**: Lines with only separators (e.g., "...", "---", etc.)
- **Other (其他)**: Other content needing deletion

---

#### 7.2.3 `deleted_report_v2.csv`

**File Path**: `/root/autodl-tmp/clean/deleted_report_v2.csv`

**File Purpose**: Records content deleted during cleaning of Western sci-fi (Western_SciFi), with detailed records of deletion reasons and locations.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `文件名` (File Name) | String | Yes | Source file name (without path) | "Blindsight.txt" |
| `行号` (Line Number) | Integer | Yes | Line number where deleted content is located (starting from 1) | 1, 2, 3, ... |
| `内容` (Content) | String | Yes | Original deleted content (complete line) | "www.feedbooks.com" |
| `原因` (Reason) | String | Yes | Deletion reason classification | "包含网址 (URL)", "目录导航", "长句高频重复", "其他" |

**Deletion Reason Type Description**:
- **Contains URL (包含网址)**: Website links (e.g., "www.feedbooks.com", "http://...")
- **Navigation (目录导航)**: Table of contents/index content (e.g., chapter navigation, page numbers, etc.)
- **Highly Repeated Long Sentences (长句高频重复)**: Sentences appearing repeatedly (e.g., the same sentence repeating 26+ times)
- **Other (其他)**: Other content needing deletion

---

### 7.3 Model Training Output File Field Description

#### 7.3.1 Model File: `best_model_epoch_X.pt`

**File Paths**:
- `training_output/models/best_model_epoch_1.pt (not saved)`
- `training_output/models/best_model_epoch_2.pt (not saved)`
- `training_output/models/best_model_epoch_3.pt (not saved)`
- `training_output_stage2_classweights/models/best_model_epoch_1.pt (not saved)`
- `training_output_stage2_classweights/models/best_model_epoch_2.pt (not saved)`
- `training_output_stage2_classweights/models/best_model_epoch_3.pt`
- `training_output_stage2_classweights_focal/models/best_model_epoch_3.pt` (**Latest Model**)

**File Purpose**: Saves the trained BERT classification model, containing model weights, optimizer state, training configuration, etc.

**File Format**: PyTorch model file (.pt)

**File Content Structure (dictionary format)**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `model_state_dict` | Dict | Yes | Model weight parameters (weights and biases for all layers) | PyTorch state_dict format |
| `optimizer_state_dict` | Dict | Yes | Optimizer state (Adam optimizer momentum, etc.) | PyTorch optimizer state_dict format |
| `epoch` | Integer | Yes | Training epoch number | 1, 2, 3 |
| `best_f1` | Float | Yes | Best F1-Score (highest F1 score on validation set) | 0.0-1.0 (e.g., 0.9998) |
| `config` | Dict | Yes | Training configuration parameters (model path, batch size, learning rate, etc.) | JSON format dictionary |

**Detailed config Field Description**:

| Field Name | Data Type | Description | Example Value |
|------------|-----------|-------------|---------------|
| `model_name` | String | Pre-trained model path | "/root/autodl-tmp/clean/bert-base-uncased-local" |
| `batch_size` | Integer | Batch size | 16 |
| `learning_rate` | Float | Learning rate | 2e-05 |
| `epochs` | Integer | Total training epochs | 3 |
| `max_length` | Integer | Maximum sequence length | 512 |
| `mode` | String | Training mode | "stage1" or "stage2" |
| `use_class_weights` | Boolean | Whether to use class weights | true or false |
| `use_focal_loss` | Boolean | Whether to use Focal Loss | true or false |
| `train_samples` | Integer | Number of training samples | 349547 |
| `val_samples` | Integer | Number of validation samples | 87387 |
| `class_names` | List | List of class names | ["Western_SciFi", "Chinese_Xianxia"] |
| `label_map` | Dict | Label mapping | {"Western_SciFi": 0, "Chinese_Xianxia": 1} |

**Recommended Usage**:
- **Best Model**: `training_output_stage2_classweights_focal/models/best_model_epoch_3.pt`
- **Performance Metrics**: Accuracy 99.99%, F1-Score 99.99%

---

#### 7.3.2 Evaluation Report File: `classification_report_*.txt`

**File Paths**:
- `training_output/reports/classification_report_best_model_epoch_3.txt`
- `training_output_stage2_classweights/reports/classification_report_best_model_epoch_3.txt`
- `training_output_stage2_classweights_focal/reports/classification_report_stage2_classweights_focal.txt`

**File Purpose**: Detailed classification performance report, containing precision, recall, F1-Score, etc. for each class.

**File Format**: Plain text (TXT)

**File Content Structure**:
```
              Precision    Recall    F1-Score    Support
Western_SciFi        0.9998     0.9999     0.9998     27529
Chinese_Xianxia      0.9999     0.9999     0.9999     59858
Accuracy                                    0.9999     87387
```

**Field Description**:

| Field Name | Data Type | Description | Range |
|------------|-----------|-------------|-------|
| `precision` | Float | Precision: the proportion of samples predicted as positive that are truly positive | 0.0-1.0 |
| `recall` | Float | Recall: the proportion of truly positive samples correctly predicted | 0.0-1.0 |
| `f1-score` | Float | F1 score: harmonic mean of precision and recall | 0.0-1.0 |
| `support` | Integer | Support: number of samples for this class in the validation set | Positive integer |

---

#### 7.3.3 Metrics File: `metrics_*.json`

**File Paths**:
- `training_output/reports/metrics_best_model_epoch_3.json`
- `training_output_stage2_classweights/reports/metrics_stage2_classweights.json`
- `training_output_stage2_classweights/reports/metrics_best_model_epoch_3.json`
- `training_output_stage2_classweights_focal/reports/metrics_stage2_classweights_focal.json`

**File Purpose**: Saves all evaluation metrics in JSON format, facilitating programmatic reading and analysis.

**File Format**: JSON

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `overall_accuracy` | Float | Yes | Overall accuracy (all classes) | 0.0-1.0 (e.g., 0.9999) |
| `roc_auc` | Float | Yes | Area under the ROC curve | 0.0-1.0 (e.g., 1.0000) |
| `pr_auc` | Float | Yes | Area under the PR curve | 0.0-1.0 (e.g., 1.0000) |
| `macro_f1` | Float | Yes | Macro-averaged F1-Score (mean of all class F1 scores) | 0.0-1.0 (e.g., 0.9998) |
| `weighted_f1` | Float | Yes | Weighted average F1-Score (weighted by sample count) | 0.0-1.0 (e.g., 0.9999) |
| `class_0` | Dict | Yes | Detailed metrics for the Western_SciFi class | See description below |
| `class_1` | Dict | Yes | Detailed metrics for the Chinese_Xianxia class | See description below |
| `confusion_matrix` | List | Yes | Confusion matrix (2x2 array) | [[TP, FN], [FP, TN]] |
| `avg_confidence` | Float | Yes | Average prediction confidence | 0.0-1.0 (e.g., 0.9990) |
| `val_loss` | Float | Yes | Validation set loss value | Positive number (e.g., 0.0005) |
| `total_training_time` | Float | No | Total training time (seconds) | Positive number (e.g., 13424.59) |
| `epoch_metrics` | List | No | Array of detailed metrics for each training epoch | Array format |

**class_0 and class_1 Field Description**:

| Field Name | Data Type | Description | Example Value |
|------------|-----------|-------------|---------------|
| `precision` | Float | Precision | 1.0 |
| `recall` | Float | Recall | 0.9992 |
| `f1` | Float | F1-Score | 0.9996 |

**confusion_matrix Description**:
- Format: `[[TP, FN], [FP, TN]]`
- `TP` (True Positive): Number correctly predicted as positive
- `FN` (False Negative): Number incorrectly predicted as negative
- `FP` (False Positive): Number incorrectly predicted as positive
- `TN` (True Negative): Number correctly predicted as negative

---

#### 7.3.4 Metrics Summary File: `metrics_summary.json`

**File Paths**:
- `training_output/reports/metrics_summary.json`
- `training_output_stage2_classweights/reports/metrics_summary.json`

**File Purpose**: Aggregates evaluation metrics from all training epochs, facilitating comparison of performance across different epochs.

**File Format**: JSON array

**Complete Field Description** (for each element in the array):

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `overall_accuracy` | Float | Yes | Overall accuracy | 0.0-1.0 (e.g., 0.9994) |
| `roc_auc` | Float | Yes | Area under the ROC curve | 0.0-1.0 (e.g., 1.0000) |
| `pr_auc` | Float | No | Area under the PR curve | 0.0-1.0 |
| `macro_f1` | Float | No | Macro-averaged F1-Score | 0.0-1.0 |
| `weighted_f1` | Float | No | Weighted average F1-Score | 0.0-1.0 |
| `model_path` | String | Yes | Model file path | "training_output/models/best_model_epoch_1.pt" |
| `epoch` | Integer | Yes | Training epoch number | 1, 2, 3 |
| `val_loss` | Float | No | Validation set loss value | Positive number |

---

#### 7.3.5 Error Samples File: `error_samples_*.csv`

**File Paths**:
- `training_output_stage2_classweights/reports/error_samples_stage2_classweights.csv`
- `training_output_stage2_classweights_focal/reports/error_samples_stage2_classweights_focal.csv`

**File Purpose**: Records samples that the model predicted incorrectly, used to analyze the model's error patterns and improvement directions.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `text` | String | Yes | Text content (complete text chunk) | "My case didn't really interest him..." |
| `true_label` | String | Yes | True label (actual class) | "Western_SciFi" or "Chinese_Xianxia" |
| `predicted_label` | String | Yes | Predicted label (model-predicted class) | "Western_SciFi" or "Chinese_Xianxia" |
| `confidence` | Float | Yes | Prediction confidence (model's confidence in the prediction) | 0.0-1.0 (e.g., 0.9998) |
| `prob_class_0` | Float | Yes | Predicted probability for class 0 (Western_SciFi) | 0.0-1.0 (e.g., 0.0001) |
| `prob_class_1` | Float | Yes | Predicted probability for class 1 (Chinese_Xianxia) | 0.0-1.0 (e.g., 0.9999) |

**Data Description**:
- Contains only incorrectly predicted samples
- Confidence is usually very high (close to 1.0), indicating the model is also "confident" about incorrect predictions
- Can be used to analyze boundary cases and model limitations
- Number of error samples is typically very small (<10)

---

### 7.4 Topic Analysis Data File Field Description

#### 7.4.1 `topic_analysis.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis.csv`

**File Purpose**: Main result file of topic modeling, containing keywords, frequency, distribution, etc. for each topic.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `topic_id` | Integer | Yes | Topic ID (-1 indicates noise/unclassified, ≥0 indicates a valid topic) | -1, 0, 1, 2, ... |
| `topic_keywords` | String | Yes | Topic keywords (comma-separated string) | "experts, space, skeleton, mental, origin..." |
| `topic_keywords_with_scores` | String | Yes | Keywords with c-TF-IDF scores (semicolon-separated) | "experts(0.2247); space(0.2234); skeleton(0.2122)..." |
| `frequency` | Integer | Yes | Number of documents in this topic | 26444, 26413, ... |
| `percentage` | Float | Yes | Document percentage of this topic (percentage) | 6.05, 6.04, ... |
| `source_distribution` | String | Yes | Source distribution (document count per category, comma-separated) | "Chinese_Xianxia: 26235, Western_SciFi: 209" |
| `files` | String | Yes | List of files containing this topic (comma-separated) | "clean_Swallowed Star.txt, clean_Foundation Trilogy.txt..." |

**Detailed Field Description**:
- **topic_id = -1**: Represents the noise topic, containing documents that cannot be classified into any topic (usually a large proportion)
- **topic_id ≥ 0**: Represents valid topics, sorted by frequency from high to low
- **topic_keywords_with_scores**: Format is "keyword(score); keyword(score); ...", higher scores indicate stronger representation of the topic
- **source_distribution**: Shows the distribution of the topic across different categories, facilitating analysis of source characteristics

---

#### 7.4.2 `topic_analysis_detailed_topics.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_detailed_topics.csv`

**File Purpose**: Detailed information about topics, including representative documents, used for in-depth understanding of the specific content of each topic.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `Topic` | Integer | Yes | Topic ID | -1, 0, 1, 2, ... |
| `Count` | Integer | Yes | Number of documents (documents contained in this topic) | 26444, 26413, ... |
| `Name` | String | Yes | Topic name (automatically generated based on keywords) | "0_experts_space_skeleton_mental" |
| `Representation` | String | Yes | Topic representation (keyword list, JSON array format) | "['experts', 'space', 'skeleton', 'mental'...]" |
| `Representative_Docs` | String | Yes | Representative documents (document chunks most representative of the topic, JSON array format) | "['Document 1...', 'Document 2...', 'Document 3...']" |

**Detailed Field Description**:
- **Representative_Docs**: Contains the document chunks most representative of the topic (usually 3), these documents can help understand the actual content of the topic
- **Representation**: List of topic keywords, stored in JSON array format
- **Name**: Topic name automatically generated based on keywords, used for quick topic identification

---

#### 7.4.3 `topic_analysis_document_topics.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_analysis_document_topics.csv`

**File Size**: ~620MB (contains the complete text of all documents)

**File Purpose**: Topic assignment table for each document, records which topic each document is assigned to, contains complete document text (for subsequent sentiment analysis).

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `document_id` | Integer | Yes | Document ID (consecutive numbering starting from 0) | 0, 1, 2, ... |
| `text_preview` | String | Yes | Text preview (first 200 characters, for quick viewing) | "Before they left, Exquisite told..." |
| `text` | String | Yes | Complete text content (complete document text) | "Complete document text..." |
| `topic_id` | Integer | Yes | Assigned topic ID | -1, 0, 1, 2, ... |
| `topic_probability` | Float | No | Topic assignment probability (confidence of the document belonging to that topic, optional) | 0.0-1.0 (e.g., 0.85) |
| `source_file` | String | Yes | Source file name | "clean_Swallowed Star.txt" |
| `source` | String | No | Source category (optional, if input data contains this field) | "Chinese_Xianxia" or "Western_SciFi" |
| `timestamp` | String | No | Timestamp (book publication/completion/award year) | "2016" |

**Data Description**:
- Each document is assigned to one topic (including noise topic -1)
- `topic_probability` represents the confidence of the document belonging to that topic (if HDBSCAN provides probability information)
- The `text` field contains the complete text; the file is large
- This is the input file for sentiment analysis

**Notes**:
- The file is very large (>200MB), containing the complete text of all documents
- It is recommended to use the `nrows` parameter to limit rows or use chunked reading
- This is the input file for sentiment analysis

---

#### 7.4.4 `topic_sentiment_final.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/topic_sentiment_final.csv`

**File Purpose**: Sentiment analysis statistical results for each topic, comprehensive result combining topic modeling and sentiment analysis, used to analyze the sentiment tendencies of different topics.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `Topic` | Integer | Yes | Topic ID | -1, 0, 1, 2, ... |
| `Sentiment_Mean` | Float | Yes | Mean sentiment score (between -1 and 1) | -1.0 to 1.0 (e.g., -0.1983) |
| `Sentiment_Std` | Float | Yes | Standard deviation of sentiment scores (reflects dispersion of sentiment distribution) | 0.0-1.0 (e.g., 0.3378) |
| `Frequency` | Integer | Yes | Number of documents in this topic | 26444, 26413, ... |
| `Positive_Count` | Integer | Yes | Number of positive sentiment documents (sentiment score > 0) | 6660, ... |
| `Negative_Count` | Integer | Yes | Number of negative sentiment documents (sentiment score < 0) | 19784, ... |
| `Neutral_Count` | Integer | Yes | Number of neutral sentiment documents (sentiment score = 0, usually 0) | 0 |
| `Keywords` | String | Yes | Topic keyword identifier (for quick topic identification) | "Topic 0" or "experts, space, skeleton" |

**Sentiment Score Description**:
- **Sentiment_Mean > 0**: Overall sentiment tendency is positive
- **Sentiment_Mean < 0**: Overall sentiment tendency is negative
- **Sentiment_Mean ≈ 0**: Overall sentiment tendency is neutral
- **Sentiment_Std**: Larger standard deviation indicates more dispersed sentiment distribution within the topic

**Sentiment Classification Criteria**:
- **Positive_Count**: Number of documents with sentiment score > 0
- **Negative_Count**: Number of documents with sentiment score < 0
- **Neutral_Count**: Number of documents with sentiment score = 0 (usually 0)

**Data Interpretation Example**:
- Topic 0: Mean sentiment -0.1983 (slightly negative), 6,660 positive documents, 19,784 negative documents
- Indicates this topic leans negative overall, but with large internal variation

---

#### 7.4.5 `temp_sentiment_docs.csv`

**File Path**: `/root/autodl-tmp/clean/bertopic_analysis/temp_sentiment_docs.csv`

**File Size**: >200MB (contains sentiment scores for all documents)

**File Purpose**: Intermediate result file for document-level sentiment scores, records the sentiment score (-1 to 1) for each document, used to generate topic-level sentiment statistics.

**File Format**: CSV

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `document_id` | Integer | Yes | Document ID (starting from 0) | 0, 1, 2, ... |
| `text` | String | Yes | Document text (complete text content) | "Complete document text..." |
| `sentiment_score` | Float | Yes | Sentiment score (between -1 and 1) | -1.0 to 1.0 (e.g., -0.2345) |
| `topic_id` | Integer | Yes | Topic ID | -1, 0, 1, 2, ... |
| `source_file` | String | Yes | Source file name | "clean_Swallowed Star.txt" |

**Sentiment Score Description**:
- **-1.0**: Extreme negative sentiment
- **0.0**: Neutral sentiment
- **1.0**: Extreme positive sentiment
- Calculated using the RoBERTa-base-sentiment model

**Notes**:
- This is an intermediate result file, usually not saved (`SAVE_TEMP_FILE = False`)
- **This file does not currently exist in the project** because the default configuration does not enable saving
- If this file is needed, set `SAVE_TEMP_FILE = True` when running sentiment analysis
- The file is large, containing the complete text of all documents

---

### 7.5 Training Configuration File Field Description

#### 7.5.1 `training_config_*.json`

**File Paths**:
- `training_output/reports/training_config_stage1.json`
- `training_output_stage2_classweights/reports/training_config_stage2_classweights.json`
- `training_output_stage2_classweights_focal/reports/training_config_stage2_classweights_focal.json`

**File Purpose**: Records the detailed parameter settings of each model training, facilitating reproduction of training results and comparison of the effects of different training configurations.

**File Format**: JSON

**Complete Field Description**:

| Field Name | Data Type | Required | Description | Range/Example |
|------------|-----------|----------|-------------|---------------|
| `model_name` | String | Yes | Pre-trained model path (local or HuggingFace model ID) | "/root/autodl-tmp/clean/bert-base-uncased-local" |
| `batch_size` | Integer | Yes | Batch size (number of samples used in each training step) | 8, 16, 32, 64 |
| `learning_rate` | Float | Yes | Learning rate (step size for model parameter updates) | 1e-05 to 5e-05 (e.g., 2e-05) |
| `epochs` | Integer | Yes | Number of training epochs (number of complete passes through the dataset) | 1, 2, 3, ... |
| `max_length` | Integer | Yes | Maximum sequence length (token count limit) | 128, 256, 512 |
| `mode` | String | Yes | Training mode identifier | "stage1" or "stage2" |
| `downsample` | Boolean | Yes | Whether to downsample (whether to downsample the majority class to balance data) | true or false |
| `use_class_weights` | Boolean | Yes | Whether to use class weights (whether to use class weights in the loss function) | true or false |
| `use_focal_loss` | Boolean | Yes | Whether to use Focal Loss (whether to use the Focal Loss loss function) | true or false |
| `train_samples` | Integer | Yes | Number of training samples (total samples in training set) | Positive integer (e.g., 349547) |
| `val_samples` | Integer | Yes | Number of validation samples (total samples in validation set) | Positive integer (e.g., 87387) |
| `class_names` | List | Yes | List of class names (names of all classes) | ["Western_SciFi", "Chinese_Xianxia"] |
| `label_map` | Dict | Yes | Label mapping (mapping from class name to numerical label) | {"Western_SciFi": 0, "Chinese_Xianxia": 1} |
| `timestamp` | String | Yes | Training timestamp (training start time) | "2026-02-12 09:31:48" |

**Training Mode Description**:
- **stage1**: First stage, basic training without class weights
- **stage2**: Second stage, uses class weights or Focal Loss for optimized training

**Recommended Configuration**:
- **Best Model Configuration**: `training_config_stage2_classweights_focal.json`
- **Features**: Uses both class weights and Focal Loss; accuracy 99.99%, F1-Score 99.99%

---
