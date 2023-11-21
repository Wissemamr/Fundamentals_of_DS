from functools import reduce
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

DEBUG: bool = False
input_data = [
    "MapReduce is a programming model",
    "It is used for processing and generating large datasets",
    "The MapReduce model involves two main steps: Map and Reduce",
]


#  Map - Extract words from each sentence
# tokenize the corpus into words
def mapper(sentence):
    words = word_tokenize(sentence)  # Tokenize the sentence
    return [(word, 1) for word in words]  # Return a list of (word, 1) pairs


# Reduce - Sum up the counts for each word
def reducer(word_counts, word_count):
    if DEBUG:
        print("--", word_counts)
    word, count = word_count
    word_counts[word] = word_counts.get(word, 0) + count
    return word_counts


# Map step
mapped_data = list(map(mapper, input_data))
flattened_data = reduce(lambda x, y: x + y, mapped_data)
word_count_result = reduce(reducer, flattened_data, {})

print("Word Count Result:")
for word, count in word_count_result.items():
    print(f"{word}: {count}")

# Calculate average length of words
total_words = sum(word_count_result.values())
total_length = sum(len(word) * count for word, count in word_count_result.items())
average_length = total_length / total_words if total_words > 0 else 0

print(f"Average Length of Words: {average_length}")
