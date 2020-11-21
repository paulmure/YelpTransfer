import json
import pickle
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from multiprocessing import Pool
from tqdm import tqdm


INPUT = os.path.join('data', 'yelp_academic_dataset_review.json')
OUTPUT = 'yelp_review_data'

MAX_SENTENCE_LEN = 30
MIN_SENTENCE_LEN = 15


def load_reveiw_from_json():
    with open(INPUT, 'r', encoding='utf-8') as f:
        data = f.readlines()
    json_data = [json.loads(entry) for entry in data]
    final_data = [(int(entry['stars']), entry['text']) for entry in json_data]
    return final_data


def get_sentences(text):
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    return sent_tokenize(cleaned_text)


def get_sentences_with_rating(entry):
    stars, text = entry
    return [(stars, sentence) for sentence in get_sentences(text)]


def get_tokens_with_ratings(entry):
    stars, sentence = entry
    tokens = word_tokenize(sentence)
    if MIN_SENTENCE_LEN <= len(tokens) <= MAX_SENTENCE_LEN:
        return stars, tokens
    return ()


def tokenize_data(data):
    with Pool() as pool:
        sentences = list(tqdm(pool.imap(get_sentences_with_rating, data)))
    sentences_flattened = [item for sublist in sentences for item in sublist]

    with Pool() as pool:
        tokenized_sentences = list(tqdm(pool.imap(get_tokens_with_ratings, sentences_flattened)))
    tokens_filtered = [entry for entry in tokenized_sentences if entry]

    return tokens_filtered


if __name__ == '__main__':
    data = load_reveiw_from_json()
    print(f'Successfully loaded {len(data):,} reviews!')
    print(f'Sample {data[-1][0]} star review: \"{data[-1][1]}\"')

    tokenized_data = tokenize_data(data)
    print(f'Successfully created {len(tokenized_data):,} tokenized data entries!')
    print(f'Sample tokenized {tokenized_data[-1][0]} star review: \"{tokenized_data[-1][1]}\"')

    print("Dumping data to disk...")
    with open(OUTPUT, 'wb') as f:
        pickle.dump(tokenized_data, f)
    print("All done!")
