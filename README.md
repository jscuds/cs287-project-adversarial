# Improving Deep Learning Model Robustness via Adversarial Attack

## Members
* Jack Scudder
* Diego Zertuche
* Moses Mayer
* Vikram Shastry


## Instructions to Run Code

### Swap Pickled Dictionary

Our attack method bases itself on `word2vec-google-news-300` embeddings. For each word on our vocabulary, we find the 10 nearest neighbors based on the cosine similarity of the embeddings, and store these 10 words with their cosine similarity score in a dictionary. You can create this dicitionary and pickle it running our `create_swap_dict.py` script, which doesn't need any modifications, and this will output a pickle file in the current directory. This script takes a long time to run (~1 hour), so we provide the pickled dictionary in our codebase - `complete_swap_word.pickle`.

### Phase 3 Base Attack

The development of our base attack is logged in the file `phase_3_baseattack.py`, which trains a simple LSTM model with a clean dataset, and a perturbed dataset. This file doesn't has to be run to get our results, but it is in the repository as a playground code. To create the dataset with the perturbed examples using our attack, the script `create_phase3_dataset.py`. This file will take the swap pickled dictionary created and create the perturbed examples for the `rotten_tomatoes` dataset. You can change the variables `swap_dict` and `dataset` to point to any created dictionary or dataset, but the code as is will take the `create_swap_dict.py` in the repo and use the Rotten Tomatoes dataset available at HuggingFace. This file will output a parquet file of the perturbed dataset into the `parquet` folder by default. We already provide the parquet file `rotten_tomatoes-PHASE3-train.parquet` in our code in the `parquet` folder.

### TextAttacks

For all other attacks (TextFooler, PWWS, DeepWord and BAE), you can use the `create_attack_datasets.py` script. You can modify the `GLOBALS` section of this file to change the dataset used, the attack used, model used or save path. By default, this script will use create a perturbed dataset of the Rotten Tomatoes dataset using the TextFooler attack, and save the parquet file of the perturbed dataset at the `parquet` folder. This scripts take quite a long time to run, ~10 hours, even on GPU, so we provide the parquet files in our `parquet` folder.

### Results

To get the results of training with different percentages of perturbed examples, you can run the scripts `BERT_results.py` and `LSTM_results.py`. Each script will take a model, dataset and attack defined in the `GLOBALS` section and read the parquet files associated to the attack and dataset from the `parquet` directory, and train the model with different percentages of the clean/perturbed dataset, which is also defined in the `GLOBALS` section. This script will output pickled dictionaries that contain the accuracies for the train, validation and test accuracies for each percentage split in the directory `results` with the following format: `{DATASET}-{ATTACK}-train.pkl`, `{DATASET}-{ATTACK}-val.pkl`, `{DATASET}-{ATTACK}-test.pkl`. Running these files as is will replicate the results from our paper.
