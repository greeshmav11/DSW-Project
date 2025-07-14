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

* `title`, `selftext`, `flair`, `author`, `num_comments`, `created_hour`, `media_type`

### Models

Two classification models (e.g., XXX, XXX) will be trained to predict `popularity_bucket`.

### Key Findings

...

---

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
│   ├── xgboost_final.ipynb
│   └── DLModelExplanations.md
│
├── notebooks/
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

1. Open `modeling.ipynb`
2. Load `cleaned_data.csv`
3. Train and evaluate your models on the `popularity_bucket` column

---

## Next Steps

* Implementing a Multimodal Model Using BERT for text fields (as it performs well for text features) and Gradient Boosting (as it handles numerical and categorical features well).
* Train models using both metadata-only and full-text features.
* Compare both models' performance across different feature sets.

---

## Team Members:

* Greeshma Vishnu (Reg. No.: 106045)
* Benecia Dsouza (Reg. No.: 106720)
* Homa Sadat Ale Ebrahim (Reg. No.: 949777)

---

