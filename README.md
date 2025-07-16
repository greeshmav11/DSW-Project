# Reddit Post Popularity Classification Project

This project involves the preprocessing and modeling of Reddit posts to predict their popularity levels. The workflow includes data acquisition (from an API), data cleaning,  and the training of two machine learning models to classify posts into popularity buckets (low, medium, high).

---

## Experimental Setup

### Dataset

* We used the **PRAW** (Python Reddit API Wrapper) library to programmatically access Reddit data through its API.
* Reddit posts were scraped from a range of subreddits, such as 'technology', 'sports', 'science', 'politics', etc.
* Data includes fields such as `title`, `selftext`, `score`, `num_comments`, `upvote_ratio`, `flair`, `author`, `created_utc`, and `url`.

### Data Cleaning

Most of the columns (or variables) contained no missing values, except for `selftext`, `flair` and `author`.

* **Handling missing values**:

1. `selftext`:
    - Since we know that an empty `selftext` means that the Reddit post contains no body text (it’s just a link post or a title-only post).
    - Hence, we filled these cells with empty string (`''`) (to show no body text).

2. `author`:
    - Similarly, we note that on Reddit, when the post and comment information is available, but not the author information, it implies that the user account was deleted (or that Reddit moderators removed the post or user). 
    - Hence we replace the NaN values with the value `[deleted]`.

3. `flair`:
    - Since "flair" is an optional tag on Reddit, not every post has a flair. This is what causes the NaN values.
    - Therefore we set the `flair` tag for these posts to `None`.


* **Timestamp**:

  * `created_utc` field was converted to `created_hour`, to retain the hour at which the post was created.
  * And then the original timestamp column was dropped (as it's then redundant)

* **URL domain analysis**:

  * Since the `url` field had a very high cardinality (almost unique for each post), to retain meaningful information, it was first grouped by url domain.
  * However, since the resulting grouping was still very sparse, we decided to group the urls to a more meaningful new field, `media_type`, which categorizes the  posts as image, video, internal Reddit link, or external link based on their urls.
  * This makes sense as these Reddit-hosted domains (like i.redd.it, v.redd.it, and www.reddit.com) for image and video hosting were most dominant.
  * The fields `url` and `url_domain` were dropped afterward.

* **Text cleaning**:

  * Basic text cleaning was performed on the two text fields: `title` and `self_text`.
  * This included:
  - Lowercasing.
  - Removing HTML tags and URLs.
  - Removing excessive whitespace.

* **Target label**:

  * Instead of predicting Reddit post scores (a regression task), we simplified the problem into a classificatiion task by categorizing the `score`field into buckets (`low`, `medium`, `high` popularity).
  * The post scores were divided into the three categories based on quantiles.
  * Since the dataset was already balanced across the `popularity_bucket` categories, we didn't need to apply any additional sampling techniques to balance the data.

### Features used

* For XGBoost model - `title`, `subreddit`, `selftext`, `flair`, `author`, `created_hour`, `media_type`, `is_self`
* For MultiLayer Perceptron model - `subreddit`, `flair`, `media_type`, `is_self`, `nsfw`, `created_hour`

### Models

- Two classification models (e.g., XGBoost, Multilayer Perceptron) were trained to predict `popularity_bucket`.
- Our models were trained on the K80 cluster
- Hyperparameter tuning was performed using Optuna


#### XGBoost Model

We trained an XGBoost classifier to predict the popularity bucket of Reddit posts based on a mix of textual and structured features.


 - The dataset was split into training and test sets with stratified sampling to keep class distribution balanced.

 - We trained the initial XGBoost model with default parameters and evaluated it on the test set, measuring accuracy, F1 score, Cohen’s Kappa, and Matthews Correlation Coefficient.

 - To improve the model, we applied cross-validation with 5 stratified folds to better estimate performance.

 - Hyperparameter tuning was performed using Optuna, which automatically searches for the best model settings to maximize the F1 score on the test set.

 - After tuning, the final model was retrained on the full training data using the best parameters found.

 - We visualized feature importance and used SHAP values to interpret how different features influenced the model’s predictions.


#### Multilayer Perceptron Model Architecture

 - This model has three hidden layers, each using ReLU activation to capture complex patterns.  
 - Dropout is added to help prevent overfitting and keep the model generalizable.   
 - For the output, we use softmax to handle multiple classes.    
 - It is trained using the Adam optimizer, with categorical crossentropy loss to make sure it effectively classifies the different categories.     
 - A Keras model is wrapped and fine-tuned with GridSearchCV to identify the optimal hyperparameters.   
 - The best-performing model is then assessed on the test set, with accuracy calculated after converting predictions and labels from one-hot encoding to class labels.    
 - We used 3-fold cross-validation to evaluate the model's performance during hyperparameter tuning.  
 - Evaluation metrics include accuracy, precision, recall, F1 score, confusion matrix, calibration curve, and Shapley values.

### Key Findings

#### XGBoost Model

- The XGBoost classifier outperformed the naive baseline across all evaluation metrics (Accuracy, F1 Score, Kappa, MCC), confirming that it successfully learned meaningful patterns in the data.

- Textual features from post titles (TF-IDF) combined with structured metadata improved the model's ability to predict popularity buckets.

- After hyperparameter tuning with Optuna, the model achieved better performance compared to the untuned version.

- 5-Fold Cross-Validation showed consistent results across all folds, indicating the model generalizes well and is not overfitting to a specific subset.

- SHAP value analysis revealed that features like subreddit, flair, and media type were the most influential in determining post popularity.

- The final model was saved and can be reused for future predictions or comparisons.

#### Multilayer Perceptron Model:
The neural network model performed significantly better using only categorical features, achieving around 56% accuracy compared to 32% when both categorical and textual features were used. This suggests that the textual data may have introduced noise or lacked sufficient signal for predicting popularity buckets.    

Our model outperforms the naive baseline by a clear margin across all metrics. This shows:    
*	It is learning useful patterns from the data.      
*	It is not guessing blindly like the baseline.    
*	Even if not perfect, it provides meaningful classification, especially in a noisy task like social media popularity prediction.  

Shapley values show the importance of features in the model:     
*   `subreddit` and `flair` are the most important features overall.    
*   `is_self` and `media_type` are moderately important.     
*   `created_hour` and `nsfw` have a very small impact, meaning they do not influence the model much.         

### Challenges faced

* **Finding Accessible Data Sources**
  - Identifying websites that were not blocked, restricted by paywalls, or limited by access policies was a key challenge during data collection.
* **Handling missing values** 
  - Required careful inspection and probing into the data and domain, to figure out the appropriate values to fill into the missing entries (for example: in the author and self-text fields).
* **Defining popularity**
  - Instead of using raw `score` values, we used quantiles to create balanced, more meaningful categories (by creating field `popularity_bucket`).
  - This was done instead of using arbitrary, fixed thresholds (which would also be dataset dependant).
* **Handling noisy URL data**
  - The `url` values were too varied and detailed for effective modeling (high cardinality).
  - Inspection of the URLs, to derive a meaningful feature such as `media_type` was a challenge.      

* **Hyperparameter Tuning Was Time-Consuming**:
  - Using Optuna to test many combinations of settings (like how deep the trees should be or how fast the model should learn) took a lot of time and computing power. We had to run over 30 trials, which needed a GPU and made the process slow.

* **Combining Different Types of Data**     
  - One challenge was merging Reddit post `titles` and `selftexts` with categorical info like `subreddit`, `flair`, `self-post status`, and `nsfw tags`. Most methods focus on just one data type, so we had to build a custom process and model that could handle both seamlessly.
* **Input Shape Mismatch & Preprocessing Errors**      
  - We faced a various errors when trying to combine text data with categorical features. It showed that making all inputs the same shape and format (through padding, encoding, and reshaping) was tricky but necessary.
* **Hyperparameter Tuning with Custom Models**   
  - Using GridSearchCV with a Keras-based model wrapped via KerasClassifier introduced further complexity, especially in passing model parameters like dropout_rate and avoiding invalid parameter errors. This tuning is straightforward for classic ML models but more error-prone with deep learning.
* **Model Architecture vs. Hyperparameters**   
  - We adjusted things like batch size, epochs, and dropout rate using GridSearch, but choosing the best model structure, such as how many layers or neurons to use, was not included. That part needs to be tested manually, and it takes a lot of time and resources. This is a common challenge in deep learning: tuning architecture is harder and often not covered in basic automatic searches.

* **Performance & Modest Accuracy**   
  - Even after combining text and other features, the final accuracy stayed low, around 32% with all features, and 57% using only the non-text ones. This shows how hard it is to predict social media popularity, as it depends on many random or hidden factors that models cannot easily capture.

---

## How to Run This Project

### Repository Structure

```
DSW PROJECT/
│
├── .venv/
│
├── cluster/
│   ├── remote-deployment.yml
│   ├── remote-job.yml
│   └── storage.yml
│
├── data/
│   ├── reddit_dataset.csv
│   └── cleaned_reddit_posts.csv
│
├── models/
│   ├── DLModel.ipynb
│   ├── DLModel_CategoricalFeatures.ipynb
│   ├── DLModel_CategoricalFeatures.py
│   ├── xgboost_final.ipynb
│   └── xgboost_final.py
│
├── preprocessing/
│   ├── reddit_post_scraper.ipynb
│   └── reddit_dataset_cleaning.ipynb
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

```

### To Clean the Data:

1. Open `reddit_dataset_cleaning.ipynb`
2. Run all cells
3. Output will be saved as `cleaned_reddit_posts.csv`

### To Train Models:

Initially, we trained the models locally using VSCode. However, due to the long training times especially during hyperparameter tuning and cross-validation, we moved the model training to Kubernetes cluster for better performance.

* Local Training (Initial Phase)
1. Open `modeling.ipynb`
2. Load `cleaned_data.csv`
3. Train and evaluate your models on the `popularity_bucket` column

* Cluster-Based Training (Final Phase)
1. Final models were trained using:
    - `models/xgboost_final.py`
    - `models/DLModel_CategoricalFeatures.py`
2. These scripts were deployed to a Kubernetes cluster using:
    - `cluster/remote-deployment.yml`
    - `cluster/remote-job.yml`
    - `cluster/storage.yml`
This setup allowed us to train faster using GPU/CPU resources available in the cluster.
---

## Future works

* Implementing a Multimodal Model Using BERT for text fields (as it performs well for text features) and Gradient Boosting (as it handles numerical and categorical features well).
* Train models using both metadata-only and full-text features.
* Compare both models' performance across different feature sets.
* Use additional tools such as Weights & Biases to keep track of training, hyperparameters and results.

---

## Team Members:

* Greeshma Vishnu (Reg. No.: 106045)
* Benecia Dsouza (Reg. No.: 106720)
* Homa Sadat Ale Ebrahim (Reg. No.: 949777)

---
