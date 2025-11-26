exec(open("random_diagram_generation.py").read())
exec(open("stats.py").read())
from tqdm import tqdm


def generate_process(probabilities, target_max_nested_xor, target_max_independent_xor, number_of_replacements, forbidden_processes):
    current_string = SEED_STRING
    
    for _ in range(number_of_replacements):
        #print("current_string:", current_string)
        current_string = replace_random_underscore(current_string, probabilities)
        replaced_string = replace_underscores(current_string)
        
     
        nested_xor = max_nested_xor(replaced_string)
        #print("nested_xor:", nested_xor)
        if nested_xor > target_max_nested_xor:
            return None  # Exit if nested_xor exceeds target
            
        independent_xor = max_independent_xor(replaced_string)
        #print("independent_xor:", independent_xor)
        if independent_xor > target_max_independent_xor:
            return None  # Exit if independent_xor exceeds target
            
        if (nested_xor == target_max_nested_xor and 
            independent_xor == target_max_independent_xor and 
            replaced_string not in forbidden_processes):
            return replaced_string
            
     
    
    # If the loop completes without finding a suitable string
    return None

def generate_multiple_processes(probabilities, target_max_nested_xor, target_max_independent_xor, 
                                number_of_replacements, forbidden_processes, num_processes, num_trials):
    generated_processes = set()
    current_forbidden = set(forbidden_processes)  # Create a copy to avoid modifying the original set

    for _ in range(num_processes):
        for _ in tqdm(range(num_trials)):
            new_process = generate_process(probabilities, target_max_nested_xor, target_max_independent_xor,
                                           number_of_replacements, current_forbidden)
            if new_process:
                generated_processes.add(new_process)
                current_forbidden.add(new_process)
                print("Generated process:", new_process)
                break  # Successfully generated a new process, move to the next one

    return generated_processes