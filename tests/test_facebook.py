from pyDOS import load_graph_from_list
import sys
sys.path.append('../')

if __name__ == '__main__':
    A = load_graph_from_list('../data/', 'facebook_combined')
    print(A)
