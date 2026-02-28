import re
import numpy as np


def find_max_task_number(string):
    # Find all occurrences of "T" followed by one or more digits
    matches = re.findall(r'T(\d+)', string)
    
    # Convert all matches to integers
    task_numbers = [int(match) for match in matches]
    
    # Return the maximum number if any matches were found, else return 0
    return max(task_numbers) if task_numbers else 0

def get_process_from_file(x, y, z):
    filename = f'dataset/generated_processes_full_{x}_{y}.txt'
    try:
        with open(filename, 'r') as file:
            for i, line in enumerate(file, 1):  # enumerate starts from 1
                if i == z:
                    return line.strip()  # strip() removes leading/trailing whitespace
        # If we've reached this point, z is greater than the number of lines in the file
        raise ValueError(f"Line {z} not found in the file. The file has fewer lines.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
def array_to_list_string(arr):
    if isinstance(arr, np.ndarray):
        return str(arr.tolist())
    elif isinstance(arr, list):
        return str(arr)
    else:
        return str(arr)
    
def pareto_to_matrix_string(pareto_set):
    if not pareto_set:
        return "[]"
    
    matrix = np.array(pareto_set).T
    
    # Convert each row (originally column) to a list of formatted strings
    string_lists = [[f"{x:.2f}" for x in row] for row in matrix]
    
    # Find the maximum width for each column
    col_widths = [max(len(row[i]) for row in string_lists) for i in range(len(string_lists[0]))]
    
    # Create the matrix string
    matrix_rows = []
    for row in string_lists:
        formatted_row = [f"{x:>{width}}" for x, width in zip(row, col_widths)]
        matrix_rows.append("" + " | ".join(formatted_row) + "")
    
    return "\\n".join(matrix_rows)  # Use \n for newline in DOT language
