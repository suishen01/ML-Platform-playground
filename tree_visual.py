from sklearn.tree import export_graphviz
import pickle

def read_list(path):
    return open(path, 'r').read().splitlines()

pkl_filename = "decisiontree_model.pkl"
# Load from file
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

feature_headers = read_list('input.txt')
label_headers = read_list('output.txt')
# Export as dot file
export_graphviz(model, out_file='tree.dot',
                feature_names = feature_headers,
                class_names = label_headers,
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
import pydot

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
