import random

# (Assuming the FirstOrderLanguageModel class from previous steps is already defined and trained)

# Modified generation method to support both modes
def generate_sentence(self, mode="sample"):
    current_token = "<START>"
    sentence = []
    
    while True:
        # Stop if we hit a dead end
        if current_token not in self.probabilities:
            break 
            
        if mode == "greedy":
            # Mode A: Deterministic / Greedy Generation
            # Always choose arg max P(w | w_previous)
            current_token = self.predict_most_probable(current_token)
        elif mode == "sample":
            # Mode B: Probabilistic Sampling
            # Sample from P(w | w_previous)
            possible_next_words = list(self.probabilities[current_token].keys())
            weights = list(self.probabilities[current_token].values())
            current_token = random.choices(possible_next_words, weights=weights, k=1)[0]
            
        if current_token == "<END>" or current_token is None:
            break
            
        sentence.append(current_token)
        
    return " ".join(sentence)

# Bind the new method to our existing class
FirstOrderLanguageModel.generate_sentence = generate_sentence

print("--- Part IX: Generating 20 Sentences (Sampling Mode) ---")
for i in range(20):
    print(f"{i+1}: {model.generate_sentence(mode='sample')}")

print("\n--- Part X: 5 Sentences (Greedy Mode) ---")
for i in range(5):
    print(f"{i+1}: {model.generate_sentence(mode='greedy')}")
    
print("\n--- Part X: 5 Sentences (Sampling Mode) ---")
for i in range(5):
    print(f"{i+1}: {model.generate_sentence(mode='sample')}")
