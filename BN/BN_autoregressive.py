import random
from collections import defaultdict

class FirstOrderLanguageModel:
    def __init__(self):
        # Nested dictionaries to hold our counts and probabilities
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probabilities = defaultdict(dict)

    def train(self, tokenised_sentences):
        # Requirement 2: Count transitions between consecutive tokens
        for sentence in tokenised_sentences:
            for i in range(len(sentence) - 1):
                current_word = sentence[i]
                next_word = sentence[i + 1]
                self.counts[current_word][next_word] += 1
                
        # Requirement 3: Construct the conditional distribution P(X_t | X_{t-1})
        for current_word, next_words in self.counts.items():
            total_following_words = sum(next_words.values())
            for next_word, count in next_words.items():
                # Probability = (Times this transition happened) / (Total transitions from current word)
                self.probabilities[current_word][next_word] = count / total_following_words

    # Requirement 4: Display the probabilities for a specified previous token
    def display_probabilities(self, previous_token):
        if previous_token in self.probabilities:
            print(f"Probabilities for '{previous_token}':")
            for next_word, prob in self.probabilities[previous_token].items():
                print(f"  - {next_word}: {prob:.4f}")
        else:
            print(f"No data for '{previous_token}'.")

    # Requirement 5: Predict the most probable next token
    def predict_most_probable(self, previous_token):
        if previous_token in self.probabilities:
            # max() finds the key (word) with the highest value (probability)
            return max(self.probabilities[previous_token], key=self.probabilities[previous_token].get)
        return None

    # Requirement 6 & 7: Generate a sentence by sampling, stopping at <END>
    def generate_sentence(self):
        current_token = "<START>"
        sentence = []
        
        while True:
            # Stop if we hit a word with no known following words
            if current_token not in self.probabilities:
                break 
                
            possible_next_words = list(self.probabilities[current_token].keys())
            weights = list(self.probabilities[current_token].values())
            
            # random.choices picks a word based on the calculated probabilities
            current_token = random.choices(possible_next_words, weights=weights, k=1)[0]
            
            # Requirement 7: Stop when <END> is generated
            if current_token == "<END>":
                break
                
            sentence.append(current_token)
            
        return " ".join(sentence)

# --- Running the Model ---

# Requirement 1: Take a list of tokenised sentences as training data
dataset = [
    ["<START>", "the", "cat", "sat", "on", "the", "mat", "<END>"],
    ["<START>", "the", "cat", "sat", "on", "the", "rug", "<END>"],
    ["<START>", "the", "dog", "sat", "on", "the", "mat", "<END>"],
    ["<START>", "the", "dog", "ran", "to", "the", "park", "<END>"],
    ["<START>", "the", "cat", "ran", "to", "the", "park", "<END>"],
    ["<START>", "the", "dog", "sat", "on", "the", "rug", "<END>"]
]

model = FirstOrderLanguageModel()
model.train(dataset)

print("--- Displaying Probabilities ---")
model.display_probabilities("cat")

print("\n--- Predicting Most Probable Next Token ---")
print(f"Next word after 'ran': {model.predict_most_probable('ran')}")

print("\n--- Generating 3 Sentences ---")
for _ in range(3):
    print(model.generate_sentence())
