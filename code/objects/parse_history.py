import networkx
import numpy as np
from bigtree import dict_to_tree,list_to_tree, tree_to_dot,preorder_iter
from bigtree.utils.plot import reingold_tilford

import matplotlib.pyplot as plt
import numpy as np


def history_figure(lhs, test_mode = False):
    if test_mode:
        n = ['t']
        x = [1]
        y = [1]

    else: 

        coord_dict = {}

        for lh in lhs:
            lh_split = lh.split('_')
            hold = "INITIAL/"
            for i in np.arange(0,int(len(lh_split)/3))*3:
                L = lh_split[i:i+3]

                if hold+L[2]+'/' in coord_dict.keys():
                    hold = hold+L[2]+'/'
                else:
                    coord_dict[hold+L[2]+'/'] = (L[1],0)
            
        cdk_dict = {str(cdk[:-1]):{'t_start':coord_dict[cdk][0]} for cdk in coord_dict.keys()}
        root = dict_to_tree(cdk_dict)
        reingold_tilford(root)
        #root.show(attr_list=["x", "y"])
        #root.show(attr_list=["t_start"])

        n = [node.name for node in preorder_iter(root)][1:]
        x = [node.get_attr('x') for node in preorder_iter(root)][1:]
        y = [node.get_attr('y') for node in preorder_iter(root)][1:]
        t = [node.get_attr('t_start') for node in preorder_iter(root)][1:]
        
        



    

    return {'x':x,'y':y,'labels':n,'t_start':t}

if __name__ == '__main__':

    test_histories = [
        'START_0_L000_',
        'START_0_L002_',
        'START_0_L003_',
        'START_0_L004_',
        'START_0_L003_START_9_L005_',
        'START_0_L000_START_23_L006_',
        'START_0_L000_START_57_L007_',
        'START_0_L000_START_57_L007_START_68_L008_']
    
    history_figure(test_histories)