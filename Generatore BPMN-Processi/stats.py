
exec(open("sese_diagram.py").read())

def max_nested_xor(expression):
    tree = PARSER.parse(expression)
    
    def _max_nested_xor(node):
        if isinstance(node, Token) or (isinstance(node, Tree) and node.data == 'task'):
            return 0
        
        if isinstance(node, Tree):
            if node.data == 'xor':
                return max(_max_nested_xor(child) for child in node.children) + 1
            elif node.data in ['sequential', 'parallel']:
                return max(_max_nested_xor(child) for child in node.children)
        
        return 0
    
    return _max_nested_xor(tree)

def max_independent_xor(expression):
    tree = PARSER.parse(expression)
    
    def _max_independent_xor(node):
        if isinstance(node, Token) or (isinstance(node, Tree) and node.data == 'task'):
            return 0
        
        if isinstance(node, Tree):
            if node.data == 'xor':
                max_child = max(_max_independent_xor(child) for child in node.children)
                return max(1, max_child)
            elif node.data in ['sequential', 'parallel']:
                return sum(_max_independent_xor(child) for child in node.children)
        
        return 0
    
    return _max_independent_xor(tree)


def max_pareto_length(tree):
    if isinstance(tree, Token):
        return len(tree.pareto) if hasattr(tree, 'pareto') else 0
    
    if isinstance(tree, Tree):
        max_length = len(tree.pareto) if hasattr(tree, 'pareto') else 0
        for child in tree.children:
            child_max = max_pareto_length(child)
            max_length = max(max_length, child_max)
        return max_length
    
    return 0

def max_theoretical_pareto_length(node):
    if isinstance(node, Token) or (isinstance(node, Tree) and node.data == 'task'):
        return 1
    
    if isinstance(node, Tree):
        if len(node.children) != 2:
            raise ValueError(f"Expected 2 children for node {node.data}, got {len(node.children)}")
        
        left_max = max_theoretical_pareto_length(node.children[0])
        right_max = max_theoretical_pareto_length(node.children[1])
        
        if node.data == 'xor':
            return left_max + right_max
        elif node.data in ['sequential', 'parallel']:
            return left_max * right_max
        else:
            raise ValueError(f"Unexpected node type: {node.data}")
    
    raise ValueError(f"Unexpected node type: {type(node)}")
