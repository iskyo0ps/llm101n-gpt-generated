import random
from collections import defaultdict,Counter

# Sample corpus
corpus = [
    "I love natural language processing",
    "I love machine learning",
    "natural language processing is fun",
    "machine learning is fascinating",
    "I love fun activities",
    "I do not love math"
]

# Tokenize the corpus into bigrams
def get_bigrams(corpus):
    bigrams = []
    for sentence in corpus:
        tokens = sentence.lower().split()
        bigrams.extend([(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
    return bigrams

bigrams = get_bigrams(corpus)
unigrams = [token for sentence in corpus for token in sentence.lower().split()]

# Count the occurrences of bigrams and unigrams
bigram_counts = Counter(bigrams)
unigram_counts = Counter(unigrams)

# Calculate bigram probabilities
bigram_probabilities = defaultdict(lambda: defaultdict(float))
for (w1, w2), count in bigram_counts.items():
    bigram_probabilities[w1][w2] = count / unigram_counts[w1]

# Generate text using the bigram model
def generate_text(start_word, num_words):
    current_word = start_word
    sentence = [current_word]
    for _ in range(num_words - 1):
        next_word_candidates = list(bigram_probabilities[current_word].keys())
        next_word_probabilities = list(bigram_probabilities[current_word].values())
        if next_word_candidates:
            current_word = random.choices(next_word_candidates, next_word_probabilities)[0]
            sentence.append(current_word)
        else:
            break
    return ' '.join(sentence)

# Example of text generation
# def main():
start_word = "language"
num_words = 100
generated_text = generate_text(start_word, num_words) 
print("Generated Text:", generated_text)

# if __name__ == "__main__":
#     main()