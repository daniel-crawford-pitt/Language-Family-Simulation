import numpy as np

from bigtree import dict_to_tree,list_to_tree, tree_to_dot,preorder_iter, hprint_tree
from bigtree.utils.plot import reingold_tilford
#root.show(attr_list=["x", "y","t_start"])
import matplotlib as pyplot

import matplotlib.pyplot as plt
import numpy as np


def history_analysis(lhs, test_mode = False):
    print('Histories: ===================================')
    for lh in lhs: print(lh)
    print('----------------------------------------------')

    history_dict = parse_histories(lhs)
    true_tree = dict_to_tree(history_dict)
    #hprint_tree(true_tree)
    adjusted_history_dict = adjust_horizon(history_dict, 40)
    adjusted_tree = dict_to_tree(adjusted_history_dict)    
    #hprint_tree(adjusted_tree)
    
    print(history_metrics(true_tree))
    print(history_metrics(adjusted_tree))


def parse_histories(lhs):
        ts = []
        tree_hists = []
        for lh in lhs:
            tree_hists.append("INITIAL/"+'/'.join([c[-4:] for c in lh.split("+")]))
            ts.append(lh[:-1].split("_")[-2])
               
        cdk_dict = {th:{'t_start':time} for th,time in zip(tree_hists,ts)}

        return cdk_dict

def adjust_horizon(cdk_dict, adjust_time):
    for v in cdk_dict.values():
        v['t_start'] = str(int(v['t_start']) - adjust_time)

    #print(cdk_dict)
    #root.show(attr_list=["t_start"])
    
    new_dict = {}
    for k,v in cdk_dict.items():
        if int(v['t_start']) > 0:
            new_dict[k] = v
        else:
            new_dict['INITIAL/'+k.split('/')[-2]] = v

    return new_dict

def history_metrics(tree):   
    
    num_lfs = len(tree.children)
    lf_sizes = [len([d for d in lf.descendants])+1 for lf in tree.children]

    return num_lfs, lf_sizes




   

    


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
