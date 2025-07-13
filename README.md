# Reddit Post Popularity Classification Project

This project involves the preprocessing and modeling of Reddit posts to predict their popularity levels. The workflow includes data acquisition (from an API), data cleaning,  and the training of two machine learning models to classify posts into popularity buckets (low, medium, high).

---

## Experimental Setup

### Dataset

* Reddit posts scraped from a range of subreddits.
* Data includes fields such as `title`, `selftext`, `score`, `num_comments`, `upvote_ratio`, `flair`, `author`, `created_utc`, and `url`.

### Preprocessing Steps

* **Handling missing values**:

  * `selftext` filled with empty string (`''`).
  * `flair` missing values filled with `'None'`.
  * `author` missing values filled with `'[deleted]'`, mimicking Reddit's own behavior.
* **Timestamp**:

  * `created_utc` converted to `created_hour` to preserve temporal information.
  * Dropped the original timestamp column.
* **Text cleaning**:

  * Lowercasing.
  * Removing HTML tags and URLs.
  * Removing excessive whitespace.
* **URL domain analysis**:

  * Extracted top domains.
  * Mapped them into `media_type`: image, video, internal Reddit, or external link.
  * Dropped `url` and `url_domain` afterward.
* **Target label**:

  * `score` bucketed into `low`, `medium`, `high` using quantiles to ensure class balance.

### Features used

* `title`, `selftext`, `flair`, `author`, `num_comments`, `created_hour`, `media_type`

### Models

Two classification models (e.g., XXX, XXX) will be trained to predict `popularity_bucket`.

---

## Key Challenges

### Challenges

* **Noisy user-generated text**: Requires careful cleaning and handling of missing/incomplete posts.
* **Imbalanced sources and post formats**: Reddit contains a wide variety of media and text formats.
* **Defining popularity**: Instead of raw scores, we use quantiles to create meaningful, balanced categories.

---

## How to Run This Project

### Repository Structure

```
.
├── data/
│   └── raw_reddit_posts.csv
├── cleaned_data.csv
├── data_cleaning.ipynb
├── modeling.ipynb
├── README.md
```

### To Clean the Data:

1. Open `reddit_data_cleaning.ipynb`
2. Run all cells
3. Output will be saved as `cleaned__reddit_posts.csv`

### To Train Models:

1. Open `modeling.ipynb`
2. Load `cleaned_data.csv`
3. Train and evaluate your models on the `popularity_bucket` column

---

## Next Steps

* Train models using both metadata-only and full-text features.
* Compare model performance across different feature sets.

---

## Team Members:

* Greeshma Vishnu (Reg. No.: 106045)
* Benecia Dsouza
* Homa Sadat Ale Ebrahim (Reg. No.: 949777)

---

