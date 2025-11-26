from lark import Lark, Tree, Token
import pydot
from PIL import Image

exec(open("utils.py").read())


PROCESS_GRAMMAR = r"""
?start: xor

?xor: parallel
    | xor "^" parallel -> xor
    | xor "^" "[" NAME "]" parallel -> xor_probability

?parallel: sequential
    | parallel "||" sequential  -> parallel

?sequential: region
    | sequential "," region -> sequential

?region: 
     | NAME   -> task
     | "(" xor ")"

%import common.CNAME -> NAME
%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""

PARSER = Lark(PROCESS_GRAMMAR, parser='lalr')

def get_tasks(t):
    return {subtree.children[0].value for subtree in t.iter_subtrees() if subtree.data == 'task'}
"""
def dot_tree(t, id=0, imp={}, pareto=False):
    if isinstance(t, Token):
        label = t.value
        impact = f", {imp[label]}" if label in imp else ""
        return f'node_{id}[label="{label}{impact}"];', id
    
    if isinstance(t, Tree):
        label = t.data
        code = ""
        last_id = id
        child_ids = []
        for i, c in enumerate(t.children):
            dot_code, last_id = dot_tree(c, last_id, imp)
            code += f'\n{dot_code}'
            child_ids.append(last_id)
            last_id += 1
        code += f'\nnode_{last_id}[label="{label}"];'
        
        for i in child_ids:
            code += f'\nnode_{last_id} -> node_{i};'
        
        return code, last_id
"""    





def dot_tree(t, id=0, imp={}, pareto=False):

    label_edge_style = 'style=dashed, arrowhead=none ,color=gray'

    if isinstance(t, Token):
        label = t.value
        impact = f", {imp[label]}" if label in imp else ""
        node_code = f'node_{id}[label="{label}{impact}"];'
        
        if pareto and hasattr(t, 'pareto'):
            pareto_id = f'{id}_pareto'
            pareto_value = pareto_to_matrix_string(t.pareto)
            pareto_code = f'node_{pareto_id}[label="{pareto_value}", shape=box, style=filled, fillcolor=lightyellow];'
            edge_code = f'node_{id} -> node_{pareto_id} [{label_edge_style}];'
            return f'{node_code}\n{pareto_code}\n{edge_code}', id
        
        return node_code, id
    
    if isinstance(t, Tree):
        label = t.data
        code = ""
        last_id = id
        child_ids = []
        for i, c in enumerate(t.children):
            dot_code, last_id = dot_tree(c, last_id, imp, pareto)
            code += f'\n{dot_code}'
            child_ids.append(last_id)
            last_id += 1
        
        node_code = f'node_{last_id}[label="{label}"];'
        code += f'\n{node_code}'
        
        for i in child_ids:
            code += f'\nnode_{last_id} -> node_{i};'
        
        if pareto and hasattr(t, 'pareto'):
            pareto_id = f'{last_id}_pareto'
            pareto_value = pareto_to_matrix_string(t.pareto)
            pareto_code = f'node_{pareto_id}[label="{pareto_value}", shape=box, style=filled, fillcolor=lightyellow];'
            edge_code = f'node_{last_id} -> node_{pareto_id} [{label_edge_style}];'
            code += f'\n{pareto_code}\n{edge_code}'
        
        return code, last_id

def print_tree(dot_code, outfile="out.png"):
    dot_string = f"digraph my_graph {{{dot_code}}}"
    #print(dot_string)  # Debugging: Print the DOT string
    graphs = pydot.graph_from_dot_data(dot_string)
    graph = graphs[0]
    graph.write_png(outfile)
    return Image.open(outfile)

PATH_IMAGE_BPMN_LARK = 'd.png'
PATH_IMAGE_BPMN_LARK_SVG ='bpmn.svg'
PATH_AUTOMATON = 'automaton.dot'
PATH_AUTOMATON_CLEANED = 'automaton_cleaned.dot'
PATH_AUTOMATON_IMAGE = 'automaton.png'
PATH_AUTOMATON_IMAGE_SVG = 'automaton.svg'
RESOLUTION = 300

def print_sese_diagram(expression, h = 0, probabilities={}, impacts={}, loop_thresholds = {}, outfile=PATH_IMAGE_BPMN_LARK, outfile_svg = PATH_IMAGE_BPMN_LARK_SVG,
                        graph_options = {}, durations = {}, names = {}, delays = {}, impacts_names = [], resolution_bpmn = RESOLUTION):
    tree = PARSER.parse(expression)
    diagram = wrap_sese_diagram(tree=tree, h=h, probabilities=probabilities, impacts=impacts, loop_thresholds=loop_thresholds, durations=durations, names=names, delays=delays, impacts_names=impacts_names)
    global_options = f'graph[ { ", ".join([k+"="+str(graph_options[k]) for k in graph_options])  } ];'
    dot_string = "digraph my_graph{ \n rankdir=LR; \n" + global_options + "\n" + diagram +"}"
    graphs = pydot.graph_from_dot_data(dot_string)
    graph = graphs[0]  
    graph.write_svg(outfile_svg)
    graph.write_svg(PATH_IMAGE_BPMN_LARK_SVG)
    graph.set('dpi', resolution_bpmn)
    graph.write_png(outfile)    
    return Image.open(outfile)
  

def dot_sese_diagram(t, id = 0, h = 0, prob={}, imp={}, loops = {}, dur = {}, imp_names = []):
    if type(t) == Token:
        label = t.value
        return dot_task(id, label, h, imp[label] if label in imp else None, dur[label] if label in dur else None, imp_names), id, id
    if type(t) == Tree:
        label = t.data
        if label == 'task':
            return dot_sese_diagram(t.children[0], id, h, prob, imp, loops, dur, imp_names)
        code = ""
        id_enter = id
        last_id = id_enter + 1
        child_ids = []
        for i, c in enumerate(t.children):
            if (label != 'natural' or i != 1)  and (label != 'loop_probability' or i !=0 ):
                dot_code, enid, exid = dot_sese_diagram(c, last_id, h, prob, imp, loops, dur, imp_names)
                code += f'\n {dot_code}'
                child_ids.append((enid, exid))
                last_id = exid + 1
        if label != "sequential":    
            id_exit = last_id
            if label == 'natural':
                code += dot_probabilistic_gateway(id_enter)
                code += dot_probabilistic_gateway(id_exit)
            elif label in {'loop', 'loop_probability'}: 
                code += dot_loop_gateway(id_enter)
                if label == 'loop':
                    code += dot_loop_gateway(id_exit)
                else:
                    code += dot_loop_gateway(id_exit)
            else: 
                label_sym = '+' if label != "xor" else "X"  
                node_label = f'[shape=diamond label="{label_sym}" style="filled" fillcolor=yellowgreen]' if label != "xor" else f'[shape=diamond label={label_sym} style="filled" fillcolor=orange]'
                code += f'\n node_{id_enter}{node_label};'
                id_exit = last_id
                code += f'\n node_{id_exit}{node_label};'
        else: 
            id_enter = child_ids[0][0]
            id_exit = child_ids[-1][1]    
        edge_labels = ['','',''] 
        if label == "natural":
            prob_key = t.children[1].value
            edge_labels = [f'{prob[prob_key] if prob_key  in prob else 0.5 }',
                           f'{round(1 - prob[prob_key], 2) if prob_key  in prob else 0.5 }']
        if label == "loop_probability":
            prob_key = t.children[0].value
            proba = loops[prob_key] if prob_key  in loops else 0.5
            edge_labels = ['',f'{proba}']
        if label != "sequential":
            for ei,i in enumerate(child_ids):
                edge_label = edge_labels[ei]
                code += f'\n node_{id_enter} -> node_{i[0]} [label="{edge_label}"];'
                code += f'\n node_{i[1]} -> node_{id_exit};'
            if label in  {'loop', 'loop_probability'}:  
                code += f'\n node_{id_exit} -> node_{id_enter} [label="{edge_labels[1]}"];'
        else:
            for ei,i in enumerate(child_ids):
                edge_label = edge_labels[ei]
                if ei != 0:
                    code += f'\n node_{child_ids[ei - 1][1]} -> node_{i[0]} [label="{edge_label}"];'              
    return code, id_enter, id_exit

def wrap_sese_diagram(tree, h = 0, probabilities={}, impacts={}, loop_thresholds = {}, durations={}, names={}, delays={}, impacts_names=[]):
    code, id_enter, id_exit = dot_sese_diagram(tree, 0, h, probabilities, impacts, loop_thresholds, durations, imp_names = impacts_names)   
    code = '\n start[label="" style="filled" shape=circle fillcolor=palegreen1]' +   '\n end[label="" style="filled" shape=doublecircle fillcolor=orangered] \n' + code
    code += f'\n start -> node_{id_enter};'
    code += f'\n node_{id_exit} -> end;'
    return code


def dot_task(id, name, h=0, imp=None, dur=None, imp_names = []):
    label = name
    impact_str = ""
    if imp is not None:
        # Always truncate impact values to two decimal places
        truncated_imp = np.round(imp, 2)
        # Only display the numerical values without labels
        impact_str = "\\n".join([f"{v:.2f}" for v in truncated_imp])
        
    label = f"{{{label}|{impact_str}}}"
    return f'\n node_{id}[label="{label}", shape=record, style="rounded,filled" fillcolor="lightblue"];'



def dot_exclusive_gateway(id, label="X"):
    return f'\n node_{id}[shape=diamond label={label} style="filled" fillcolor=orange];'

def dot_probabilistic_gateway(id, label="N"):
    return f'\n node_{id}[shape=diamond label={label} style="filled" fillcolor=orange];' 

def dot_loop_gateway(id, label="X"):
    return f'\n node_{id}[shape=diamond label={label} style="filled" fillcolor=yellow];' 

def dot_parallel_gateway(id, label="+"):
    return f'\n node_{id}[shape=diamond label={label} style="filled" fillcolor=yellowgreen];'

def dot_rectangle_node(id, label):
    return f'\n node_{id}[shape=rectangle label={label}];' 