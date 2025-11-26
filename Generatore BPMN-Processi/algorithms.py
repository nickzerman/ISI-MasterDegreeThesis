import numpy as np
from lark import Tree, Token

def build_max_values(node, impacts):
    if isinstance(node, Token) or (isinstance(node, Tree) and node.data == 'task'):
        task_name = node.value if isinstance(node, Token) else node.children[0].value
        node.max_values = np.array(impacts[task_name])
        return node.max_values
    
    if isinstance(node, Tree):
        if len(node.children) != 2:
            raise ValueError(f"Expected 2 children for node {node.data}, got {len(node.children)}")
        
        left_max = build_max_values(node.children[0], impacts)
        right_max = build_max_values(node.children[1], impacts)
        
        if node.data in ['sequential', 'parallel']:
            node.max_values = left_max + right_max
        elif node.data == 'xor':
            node.max_values = np.maximum(left_max, right_max)
        else:
            raise ValueError(f"Unexpected node type: {node.data}")
        
        return node.max_values
    
    raise ValueError(f"Unexpected node type: {type(node)}")

def is_dominated(v, v_prime):
    return np.all(v <= v_prime) and np.any(v < v_prime)

def enforce_incomparability(vectors):
    pareto_set = []
    for v in vectors:
        if not any(is_dominated(v_prime, v) for v_prime in vectors if not np.array_equal(v, v_prime)):
            if not any(np.array_equal(v, v_prime) for v_prime in pareto_set):
                pareto_set.append(v)
    return pareto_set

def build_pareto_frontier(node, impacts):
    if isinstance(node, Tree):
        if node.data == 'task':
            task_name = node.children[0].value
            node.pareto = [np.array(impacts[task_name])]
        elif node.data in ['sequential', 'parallel']:
            build_pareto_frontier(node.children[0], impacts)
            build_pareto_frontier(node.children[1], impacts)
            node.pareto = [v1 + v2 for v1 in node.children[0].pareto for v2 in node.children[1].pareto]
            node.pareto = enforce_incomparability(node.pareto)
        elif node.data == 'xor':
            build_pareto_frontier(node.children[0], impacts)
            build_pareto_frontier(node.children[1], impacts)
            node.pareto = node.children[0].pareto + node.children[1].pareto
            node.pareto = enforce_incomparability(node.pareto)
        else:
            raise ValueError(f"Unknown node type: {node.data}")
    else:
        raise ValueError(f"Expected Tree node, got {type(node)}")