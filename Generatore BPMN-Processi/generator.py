'''
Materiale importato dalla libreria di riferimento del Prof. Sala - process-impact-benchmarks
'''

from random_diagram_generation import SEED_STRING, replace_random_underscore, replace_underscores
from sese_diagram import PARSER, print_sese_diagram, print_tree, dot_tree
from stats import max_nested_xor, max_independent_xor
from lark import Lark, Tree, Token

# Generate a process
current_string = SEED_STRING
probabilities = 0.40,0.20,0.40 #xor, parallel, seq
print(f"Using probabilities: {probabilities}")

iterations = 4 #Quante task diverse
for _ in range(iterations):
    current_string = replace_random_underscore(current_string, probabilities)

process = replace_underscores(current_string) #Processo effettivo
print("Generated Process:")
print(process)

# Parse the process - prendo il processo
tree = PARSER.parse(process)
print(tree)
print(tree.data) #'Nome' del nodo
print(tree.children) #I due figli dell'albero

for i, c in enumerate(tree.children):
    print(i, c)

print(dot_tree(tree)[0])


# Print process statistics
print(f"\nMax Nested XOR: {max_nested_xor(process)}")
print(f"Max Independent XOR: {max_independent_xor(process)}")

# Visualize the process
print("\nProcess Diagram:")
print_sese_diagram(process)