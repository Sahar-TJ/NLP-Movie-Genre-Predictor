import argparse
import json
import os
import tensorflow as tf
import pickle
import re
import nltk
import lime
from lime import lime_text

def load_model_and_lookup():
    model = tf.keras.models.load_model('model')
    with open('categories_forward_lookup.pkl', 'rb') as f:
        categories_forward_lookup = pickle.load(f)
    return model, categories_forward_lookup

def read_description(file_path):
    with open(file_path, 'r') as f:
        description = f.read()
    return description

def predict_genre(model, categories_forward_lookup, description):
    predictions = model.predict([description])[0]
    genres = {genre: predictions[i] for genre, i in categories_forward_lookup.items()}
    sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
    return genres, sorted_genres[:3]

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def preprocess_description(description):
    description = clean_text(description)
    description = remove_stopwords(description)
    return description

def predict_fn(texts, model):
    descriptions = [preprocess_description(description) for description in texts]
    predictions = model.predict(descriptions)
    return predictions

def classify_tv_show(input_file, output_json_file, encoding='UTF-8', explanation_output_dir=None):
    try:
        # Read the description from the input file
        description = read_description(input_file)
        
        # Load your model
        model, categories_forward_lookup = load_model_and_lookup()
        
        # Implement your classification logic
        _, top_genres = predict_genre(model, categories_forward_lookup, description)
        genres_with_probabilities = {genre[0]: float("{:.2f}".format(genre[1])) for genre in top_genres}

        # Write the identified genres with their probabilities to the output JSON file
        with open(output_json_file, 'w', encoding=encoding) as json_file:
            json.dump(genres_with_probabilities, json_file, ensure_ascii=False)

        # Optionally, write an explanation to the explanation output directory
        if explanation_output_dir:
            class_names = list(categories_forward_lookup.keys())
            explainer = lime_text.LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(description, lambda x: predict_fn(x, model), num_features=10)
            explanation_filename = os.path.join(explanation_output_dir, "explanation.html")
            exp.save_to_file(explanation_filename)
           
            # with open(explanation_filename, 'w', encoding=encoding) as exp_file:
                
            #     for item in exp.as_list():
            #         exp_file.write(str(item) + '\n')
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()

    classify_tv_show(args.input_file, args.output_json_file, args.encoding, args.explanation_output_dir)
