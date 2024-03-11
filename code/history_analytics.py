import numpy as np

from bigtree import dict_to_tree,list_to_tree, tree_to_dot,preorder_iter, hprint_tree
from bigtree.utils.plot import reingold_tilford

import matplotlib as pyplot

import matplotlib.pyplot as plt
import numpy as np


def history_analysis(lhs, test_mode = False):
    print('Histories: ===================================')
    for lh in lhs: print(lh)
    print('----------------------------------------------')

    if test_mode:
        n = ['t']
        x = [1]
        y = [1]

    else: 

        ts = []
        tree_hists = []
        for lh in lhs:
            tree_hists.append("INITIAL/"+'/'.join([c[-4:] for c in lh.split("+")]))
            ts.append(lh[:-1].split("_")[-2])
        
        
        cdk_dict = {th:{'t_start':time} for th,time in zip(tree_hists,ts)}

        #print(cdk_dict)
        root = dict_to_tree(cdk_dict)
        #reingold_tilford(root)
        #root.show(attr_list=["x", "y","t_start"])
        root.show(attr_list=["t_start"])
        #hprint_tree(root)

        


if __name__ == "__main__":
    

    test_histories = [
        'START_0_L001+',
'START_0_L002+START_59_L011+',
'START_0_L003+START_14_L007+',
'START_0_L004+',
'START_0_L005+START_51_L009+',
'START_0_L003+START_14_L006+',
'START_0_L005+START_51_L008+START_62_L013+',
'START_0_L002+START_59_L010+',
'START_0_L005+START_51_L008+START_62_L012+',
]
    

    history_analysis(test_histories)
