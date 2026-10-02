print("--- Testing Probability Normalisation ---")
for word, next_words in model.probabilities.items():
    # Sum all the probabilities of the words following the current 'word'
    total = sum(next_words.values())
    print(f"Total probability for '{word}': {total:.2f}")
    
    # In a perfect model, this should always print 1.00
