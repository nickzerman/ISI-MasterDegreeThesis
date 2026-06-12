import re
from lark import Tree
import torch
from pm4py.objects.process_tree.obj import Operator

def createNAryTree(tree):
    """
        From binary tree to n-ary tree

        Parameters
        ----------
        tree
            binary tree

        Returns
        ----------
        tree
            n-ary tree
    """
    if tree.data == 'task': #CASO BASE: se il tree è una task torno direttamente la task
        return tree

    children_processed = [createNAryTree(child) for child in tree.children] #Processo ogni figlio del nodo (saranno al massimo due visto che l'albero in input è nario)

    children = []
    for i,child in enumerate(children_processed): #Ciclo i figli processati
        #if child.data == tree.data and not(child.data=="loop" and i!=0): #Se il tipo di nodo è lo stesso elimino il nodo e copio i figli
        if child.data == tree.data:
            children.extend(child.children)
        else: #Altrimenti copio tutto il sottoalbero col tipo di nodo diverso
            children.append(child)

    return Tree(tree.data, children) #Ritorno l'albero n-ario

# Mapping attività → codice T01, T02, ...
# Le attività che iniziano con 'R' (Release, Return) colliderebbero con i nomi delle regioni
# (R0, R1, ...) in PetriNetP, che distingue regioni da task tramite il primo carattere.
def get_activities_from_tree(node):
    if node.operator is None:
        return [node.label] if node.label is not None else []
    return [a for child in node.children for a in get_activities_from_tree(child)]

# Conversione ProcessTree (pm4py) → stringa SESE del progetto
def process_tree_to_sese(node, mapping):
    """
    Converte ricorsivamente un ProcessTree pm4py nella stringa SESE.
    I nodi tau (silenziosi) vengono ignorati.
    Nel loop pm4py *(body, redo), si prende solo il corpo (primo figlio).
    Il wrapping tra parentesi garantisce il parsing corretto della grammatica.
    """
    if node.operator is None:
        if node.label is None:
            return None  # nodo tau: ignorato
        return mapping[node.label]

    children_sese = [
        s for child in node.children
        if (s := process_tree_to_sese(child, mapping)) is not None
    ]

    if not children_sese:
        return None
    if len(children_sese) == 1:
        return f'->({children_sese[0]})' if node.operator == Operator.LOOP else children_sese[0]

    def wrap(s):
        return s if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s) else f'({s})'

    if node.operator == Operator.SEQUENCE:
        return ','.join(wrap(c) for c in children_sese)
    elif node.operator == Operator.XOR:
        return '^'.join(wrap(c) for c in children_sese)
    elif node.operator == Operator.PARALLEL:
        return '||'.join(wrap(c) for c in children_sese)
    elif node.operator == Operator.LOOP:
        return f'->({children_sese[0]})'
    return None