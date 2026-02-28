import random

def count_underscores(string):
    return string.count('_')

def replace_underscores(input_string):
    count = 0
    result = ""
    for char in input_string:
        if char == '_':
            count += 1
            result += f"T{count}"
        else:
            result += char
    return result

SEED_STRING = '_'

def guess_three_numbers():
    # Generate two random numbers between 0 and 1
    a = random.random()
    b = random.random()

    # Ensure a <= b
    if a > b:
        a, b = b, a

    # Calculate the three numbers
    x = a
    y = b - a
    z = 1 - b

    return x, y, z

def weighted_choice(choices, probabilities):
    if len(choices) != 3 or len(probabilities) != 3:
        raise ValueError("Must provide exactly 3 choices and 3 probabilities")
    
    if not abs(sum(probabilities) - 1.0) < 1e-6:
        raise ValueError("Probabilities must sum to 1")
    
    r = random.random()
    
    if r < probabilities[0]:
        return choices[0]
    elif r < probabilities[0] + probabilities[1]:
        return choices[1]
    else:
        return choices[2]

def replace_random_underscore(input_string, probabilities= None):
    replacements = ["(_ ^ _)", "(_ || _)", "(_ , _)"]
    
    # Find all underscore positions
    underscore_positions = [i for i, char in enumerate(input_string) if char == '_']
    
    # If no underscores, return the original string
    if not underscore_positions:
        return input_string
    
    # Choose a random underscore position
    random_position = random.choice(underscore_positions) 
    
    # Choose a random replacement
    random_replacement = random.choice(replacements) if probabilities is None else weighted_choice(replacements, probabilities)
    
    # Replace the chosen underscore with the chosen replacement
    return input_string[:random_position] + random_replacement + input_string[random_position+1:]


