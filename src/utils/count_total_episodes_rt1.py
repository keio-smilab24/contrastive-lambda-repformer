import os
import json

def count_episodes(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.startswith("episode"):
                count += len(os.listdir(os.path.join(root, dir))) / 2
    return int(count)

def count_rates(directory):
    true_num = 0
    false_num = 0
    total = 0
    with open(os.path.join("data/RT-1/info.json")) as f:
        info = json.load(f)
    for episode in os.listdir(directory):
        total += 1
        if info[episode]["succeeded"]:
            true_num += 1
        else:
            false_num += 1

    return true_num, false_num, total

def calculate_dataset_metrics(info):
    # Initialize variables to calculate vocabulary size, average sentence length, and total word count
    vocabulary = set()
    total_words = 0
    total_sentences = 0

    # Process each episode's description
    for episode in info.values():
        # Tokenize the description into words
        words = episode['description'].split()
        total_words += len(words)
        total_sentences += 1
        vocabulary.update(words)

    # Calculate vocabulary size and average sentence length
    vocabulary_size = len(vocabulary)
    average_sentence_length = total_words / total_sentences

    return vocabulary_size, average_sentence_length, total_words

train_dir = "data/RT-1/images/train"
valid_dir = "data/RT-1/images/valid"
test_dir = "data/RT-1/images/test"
with open(os.path.join("data/RT-1/info.json")) as f:
    info = json.load(f)

dirs = [train_dir, valid_dir, test_dir]
vocabulary_size, average_sentence_length, total_words = calculate_dataset_metrics(info)
print("Average Sentence Length:", average_sentence_length)
print("Vocabulary Size:", vocabulary_size)
print("Total Word Count:", total_words)

train_episodes = count_episodes(train_dir)
valid_episodes = count_episodes(valid_dir)
test_episodes = count_episodes(test_dir)

train_true, train_false, total = count_rates(train_dir)
test_true, test_false, total = count_rates(test_dir)

print("Train episodes:", train_episodes)
print("Valid episodes:", valid_episodes)
print("Test episodes:", test_episodes)
print("Total episodes:", train_episodes + valid_episodes + test_episodes)
print("\n")
print("Train true:", train_true)
print("Train false:", train_false)
print("\n")
print("Test true:", test_true)
print("Test false:", test_false)
