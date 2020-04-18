import sys
sys.path.append('../')
from pyDOS import load_graph_from_mat

if __name__ == '__main__':
    A = load_graph_from_mat('../data/','erdos02-cc')
    print(A)
