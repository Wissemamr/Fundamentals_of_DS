# The implementation of entropy
import math
import random

def H(sentece : str) -> float :
    """
    Equation 3.49 (Shannon's Entropy) is implemented.
    """
    entropy = 0
    # in 8 bits there are 2^8 = 265 possible ASCII characters
    for char in range(256):
        prob_x = sentece.count(chr(char)) / len(sentece)
        if prob_x > 0:
            entropy += - prob_x * math.log2(prob_x)
            
    return entropy

if __name__ == "__main__":
    message1 = "hello"
    message2 ="Hello darkness my old friend, i came to talk with you again, because a vision softly creeping , left its seeds while I was sleeping"
    print((f"Entropy of message1: {H(message1)}"))
    print(f'Entropy of message2: {H(message2)}')
    
    
    # The entropy of the more complex message is higher because it has more characters and the probability of each character is lower,the uncertainty of a character c is higher, the distribution is more disperse.