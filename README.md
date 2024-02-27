# NLP-Movie-Genre-Predictor


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
