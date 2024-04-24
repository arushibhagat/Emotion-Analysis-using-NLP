# Emotion-Analysis-using-NLP

*Introduction*

In an era marked by unprecedented digital communication, understanding human emotions through natural language has become pivotal. This project aims to delve into the realm of Emotion Analysis using Natural Language Processing (NLP). By harnessing advanced computational techniques, we seek to unravel the intricate nuances of human sentiment expressed in textual data. Through this endeavor, we aspire to equip machines with the ability to comprehend and respond to human emotions effectively. Leveraging state-of-the-art NLP algorithms, we endeavor to develop robust models capable of accurately detecting and categorizing emotions expressed in diverse textual contexts. Ultimately, this project aims to contribute to fields such as sentiment analysis, customer feedback analysis, mental health monitoring, and personalized user experiences. Through interdisciplinary collaboration between linguistics, psychology, and computer science, we embark on a journey to decipher the language of emotions in the digi*tal age.


*Problem Statement*

The project aims to develop a system capable of accurately detecting and analyzing emotions from text data using Natural Language Processing (NLP) techniques. This involves identifying and categorizing emotions expressed in written content, such as social media posts, emails, or reviews. The goal is to create a robust model that can discern subtle nuances in emotion, considering factors like context, tone, and language nuances. The challenge lies in training a model that can effectively capture the complexity and variability of human emotions expressed through language.


*Dataset description*

The dataset used in this study comprises of texts and their emotion labels sourced from Kaggle. Its enormous size ensures that accuracy meets the bench mark.
Kaggle Emotion Dataset for Emotion Recognition Tasks:

•	.csv format
•	Size (1.5-2 MB)
•	https://www.kaggle.com/datasets/parulpandey/emotion-dataset


*Model Building*

1.	Data Preprocessing
•	Data Cleaning: HTML tags were removed from the text, and the text was converted to lowercase.
•	Tokenization: Text was tokenized into words.
•	Punctuation and Special Characters: Removed from the tokens.
•	Contractions Handling: Expanded contractions (e.g., "don't" to "do not").
•	Stopwords Removal: Common English stopwords were removed.
•	Lemmatization: Words were lemmatized to their base forms.

2.	Exploratory Data Analysis (EDA)
•	Label Distribution: Analyzed the distribution of emotions in the dataset.
•	Word Clouds: Created word clouds for each emotion to visualize common words associated with each emotion.
•	N-gram Analysis: Investigated the distribution of n-grams (sequences of n words) within the text data.

3.	Feature Extraction
•	TF-IDF Vectorization: Converted preprocessed text into TF-IDF vectors for model training.

4.	Model Selection and Training
•	Trained multiple classifiers including Logistic Regression, Decision Tree, Random Forest, K Nearest Neighbors, SVM, and Naive Bayes.
•	Split the data into training and testing sets (80/20 ratio).
•	Evaluated models using classification metrics like precision, recall, and F1-score.

5.	Model Evaluation
•	Evaluated each model's performance on the test set using classification reports.
•	Identified the best-performing models based on accuracy and other metrics.

6.	Hyperparameter Tuning
•	Conducted hyperparameter tuning using GridSearchCV for selected classifiers to optimize model performance.

7.	Model Saving
•	Saved the trained model using joblib for potential deployment.

8.	Flask App for user input
•	It can take text input from user and predict the emotion related to it.


Libraries Used:
NumPy: Essential for numerical operations and data manipulation.
NLTK: Supports text preprocessing and sentiment analysis for emotion tasks.
Pandas: Facilitates efficient data handling and preprocessing.
Matplotlib: Enables visualization of model performance and emotional trends.
Scikit-Learn: Supports predictive data analysis of dataset.


*Results*

Metrics such as accuracy, precision, recall, and F1-score were used to assess model performance. Logistic Regression exhibited the highest accuracy and balanced performance across emotion categories.
After using GridSearchCV for Hyperparameter Tuning, the hyperparameters for each classifier were optimized as follows:

Logistic Regression: {'C': 10}
Decision Tree: {'max_depth': None, 'min_samples_split': 10}
Random Forest: {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 300}
K Nearest Neighbors: {'n_neighbors': 3, 'weights': 'distance'}
Support Vector Machine: {'C': 1, 'kernel': 'linear'}
Naive Bayes: No hyperparameters to tune


*Conclusion*

In conclusion, the journey through the realm of emotion analysis using Natural Language Processing (NLP) has been both insightful and rewarding. Through the lens of advanced computational techniques, we've delved into the intricate nuances of human emotions expressed through text, uncovering patterns, sentiments, and underlying meanings.

The application of logistic regression, a powerful machine learning algorithm, has played a pivotal role in our endeavor. With an impressive accuracy of 86%, it has demonstrated its efficacy in accurately categorizing and predicting emotions from textual data. This success underscores the potential of leveraging NLP and machine learning methodologies in understanding and interpreting human emotions at scale.

In essence, this project serves as a testament to the transformative power of NLP in decoding the intricate tapestry of human emotions. With each advancement, we inch closer to a deeper understanding of ourselves and the world around us, paving the way for a more empathetic and interconnected future.

