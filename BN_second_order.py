import random
from collections import defaultdict

class SecondOrderLanguageModel:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probabilities = defaultdict(dict)

    def train(self, tokenised_sentences):
        # Count transitions of observed triples (Word 1 + Word 2 -> Word 3)
        for sentence in tokenised_sentences:
            for i in range(len(sentence) - 2):
                context = (sentence[i], sentence[i + 1])
                next_word = sentence[i + 2]
                self.counts[context][next_word] += 1
                
        # Construct the conditional distribution P(X_t | X_{t-2}, X_{t-1})
        for context, next_words in self.counts.items():
            total = sum(next_words.values())
            for next_word, count in next_words.items():
                self.probabilities[context][next_word] = count / total

    def generate_sentence(self):
        # We need two starting words to kick off the prediction
        # Let's randomly pick an observed starting pair that begins with <START>
        starting_pairs = [k for k in self.probabilities.keys() if k[0] == "<START>"]
        if not starting_pairs:
            return ""
            
        current_context = random.choice(starting_pairs)
        sentence = list(current_context)
        
        while True:
            if current_context not in self.probabilities:
                break 
                
            possible_next_words = list(self.probabilities[current_context].keys())
            weights = list(self.probabilities[current_context].values())
            next_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            
            if next_word == "<END>":
                break
                
            sentence.append(next_word)
            # Shift the context window forward
            current_context = (current_context[1], next_word)
            
        return " ".join(sentence)

# Testing the second order model
model2 = SecondOrderLanguageModel()
model2.train(dataset) # Using the dataset from earlier
print("Second-Order Generated Sentence:")
print(model2.generate_sentence())
