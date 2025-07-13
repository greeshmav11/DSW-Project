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




