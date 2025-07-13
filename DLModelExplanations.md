## Project Summary: Reddit Post Popularity Prediction Using MLP    

In this project, we developed a deep learning classification model using Keras and TensorFlow to predict the popularity level (popularity_bucket) of Reddit posts.    
The model was based on a Multilayer Perceptron (MLP) architecture and utilized both textual data (from the post's title and selftext) and categorical metadata (such as subreddit, flair, is_self, and nsfw).  

 ### Data Processing
•	Text features: Tokenized, padded, and passed through an embedding layer followed by global average pooling.   
•	Categorical features: Encoded using label encoding.    
•	Both feature types were combined into a single input vector for the model.    
Model Architecture     
•	Embedding layer + global average pooling for text input.     
•	Dense (fully connected) layer combining text and categorical features.     
•	Dropout layer to reduce overfitting.     
•	Final softmax layer for multi-class classification.    

### Results    
•	Best hyperparameters found:     
  o	Batch size: 16     
  o	Epochs: 15     
  o	Dropout rate: 0.3     
•	Best cross-validated accuracy: 0.3440      
•	Test set accuracy on unseen data: 0.3234      
•	Test loss: 1.0993      
Despite modest accuracy, this project successfully demonstrated a full pipeline of combining textual and categorical data, building a deep learning model, and applying hyperparameter tuning.       

After removing the textual features and training the deep learning model solely on categorical and numerical metadata (such as subreddit, flair, media_type, is_self, nsfw, and created_hour), the model achieved a significantly higher performance.     
The best cross-validated accuracy improved to 0.5696, and the final test accuracy reached 0.5786, compared to much lower performance when combining with text.     
This suggests that, in this case, the categorical features were more informative and effective for predicting post popularity than the noisy or sparse textual content.        

### Results     
•	Best hyperparameters found:    
  o	Batch size: 16     
  o	Epochs: 20     
  o	Dropout rate: 0.3      
•	Best cross-validated accuracy: 0.5679      
•	Test set accuracy on unseen data: 0.5647     
•	Test loss: 0.9387    

### Naive baseline     
These results show why building a proper model is important and how it performs significantly better than a naive approach:     

#### Model Performance Metrics 	
Accuracy:  0.5617   
Precision: 0.6384   
Recall:    0.5617    
F1 Score:  0.5395	     

#### Naive Baseline Metrics    
Accuracy: 0.3234   
Precision: 0.1046    
Recall: 0.3234    
F1 Score: 0.1580    

Our model outperforms the naive baseline by a clear margin across all metrics. This shows:    
•	It is learning useful patterns from the data.   
•	It is not guessing blindly like the baseline.    
•	Even if not perfect, it provides meaningful classification, especially in a noisy task like social media popularity prediction.   


### Challenges Faced in the Project:   
#### Combining Different Types of Data:    
One of the biggest challenges we faced was figuring out how to bring together two very different kinds of information: the text from Reddit post titles and selftexts, and the categorical details like subreddit, flair, whether the post is a self-post, or marked as NSFW. Most existing methods usually handle just one type of data, so we had to create a custom process and design a model that could work smoothly with both at the same time.    

#### Input Shape Mismatch & Preprocessing Errors:     
We faced errors like “inconsistent number of samples” or “setting an array element with a sequence” when trying to combine text data with categorical features. It showed that making all inputs the same shape and format (through padding, encoding, and reshaping) was tricky but necessary.       

#### Hyperparameter Tuning with Custom Models:        
Using GridSearchCV with a Keras-based model wrapped via KerasClassifier introduced further complexity, especially in passing model parameters like dropout_rate and avoiding invalid parameter errors. This tuning is straightforward for classic ML models but more error-prone with deep learning.    

#### Model Architecture vs. Hyperparameters:     
We adjusted things like batch size, epochs, and dropout rate using GridSearch, but choosing the best model structure, like how many layers or neurons to use, wasn’t included. That part needs to be tested manually, and it takes a lot of time and resources. This is a common challenge in deep learning: tuning architecture is harder and often not covered in basic automatic searches.     

#### Performance & Modest Accuracy:      
Even after combining text and other features, the final accuracy stayed low, around 32% with all features, and 57% using only the non-text ones. This shows how hard it is to predict social media popularity. Other research also finds that it's tricky because popularity depends on many random or hidden factors that models can’t easily capture.     












