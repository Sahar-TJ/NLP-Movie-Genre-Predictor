# GenreSmart: AI-Powered Movie Classifier and Recommender


### Steps to run the files:

__First activate the environment in the kernel:__
>myenv\Scripts\activate

__Run the training file__
>python train.py --training-data tvmaze.sqlite

__Run the classify file__
>python classify.py --input-file desc.txt --output-json-file output.json --encoding UTF-8 --explanation-output-dir explanation

Check the explanation.html for lime output and output.json for the genre output

__Run the index file__
>python index.py --raw-data tvmaze.sqlite

__Run the Search file__
>python search.py --input-file desc.txt --output-json-file outputs.json --index-file indexed_data.pkl --encoding UTF-8

Check the outputs.json for the similar show outputs. 


## Files

### train.py
The codet is designed to train a neural network model using TV show descriptions and their associated genres. It involves several key steps:

- __Data Loading:__ The script begins by establishing a connection to an SQLite database to fetch data from tables containing TV show descriptions and genres.
- __Data Cleaning and Preprocessing:__ It applies text preprocessing techniques to clean the descriptions, including removing stopwords, HTML tags, URLs, and lemmatizing the text to reduce words to their base forms.
- __Data Preparation:__ The script prepares the target variable by one-hot encoding the genres and splits the data into training and validation sets.
- __Model Training:__ A neural network model is trained using the preprocessed text data. The model architecture includes a TextVectorization layer to convert text into token vectors, an Embedding layer for word embeddings, and Dense layers for classification.
- __Evaluation:__ The trained model is evaluated on a test set to determine its accuracy.
- __Saving the Model:__ The final model and the mapping between genres and their respective indices in the output layer are saved for later use in predictions.


## classify.py
The code used to classify the genre of a TV show based on its description using the previously trained model. The main activities include:

- __Model Loading:__ It loads the trained model and the genre-index mapping.
- __Data Preprocessing:__ The TV show description is cleaned and preprocessed using the same methods as in the training script.
- __Prediction:__ The script predicts the genres of the TV show and ranks them based on their probabilities.
- __Output:__ The predicted genres and their probabilities are saved to a JSON file. Optionally, an explanation of the prediction can be generated using LIME and saved as an HTML file.


## index.py
The code is intended to index TV show data for efficient retrieval:

- __Data Extraction:__ Connects to the SQLite database to fetch TV show descriptions.
- __Preprocessing:__ Applies text preprocessing on the descriptions, similar to the training script.
- __Indexing:__ Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert the preprocessed descriptions into a matrix of TF-IDF features.
- __Saving Indexed Data:__ The indexed data, along with the TF-IDF matrix and the vectorizer, are saved to a file for future use in searching or information retrieval tasks.


## search.py
The code is tailored for searching and retrieving TV shows based on a textual query, leveraging the indexed data created by index.py. Its main functions include:

- __Index Loading:__ It begins by loading the indexed data, which includes the TF-IDF matrix and the trained vectorizer, essential for understanding the importance and frequency of terms within the dataset.
- __Query Preprocessing:__ The user's search query is cleaned and preprocessed to match the format of the indexed data, ensuring consistency in text representation.
- __Query Vectorization:__ The preprocessed query is transformed into a TF-IDF vector using the loaded vectorizer, aligning it with the feature space of the indexed TV show descriptions.
- __Similarity Calculation:__ It computes the cosine similarity between the query vector and the indexed TF-IDF vectors, identifying the most relevant TV shows based on textual similarity.
- __Ranking and Retrieval:__ Based on the cosine similarity scores, TV shows are ranked, and the top matches are selected. The most relevant TV shows that best match the query are displayed.
- __Output:__ The matched TV shows, along with their similarity scores or other relevant information (like titles or IDs), are saved to a JSON file. This file serves as the output, providing users with a list of TV shows that closely align with the search query.






