import numpy as np
import re
from bigtree import dict_to_tree,list_to_tree, tree_to_dot,preorder_iter, hprint_tree, print_tree
from bigtree.utils.plot import reingold_tilford
#root.show(attr_list=["x", "y","t_start"])
import matplotlib as pyplot
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def history_analysis(lhs, output_file, test_mode = False):
    #print('Histories: ===================================')
    #for lh in sorted(lhs): print(lh)
    #print('----------------------------------------------')
    #for lh in sorted(lhs):
    #    print(lh)
    #print(all_lang_histories,"\n ======== \n",alive_language_histories)

    all_lang_histories, alive_language_histories = sep_alive_langs(lhs)
    
    #Parse Histories
    print("TRUE TREE")
    all_history_dict = parse_histories(all_lang_histories)
    all_history_dict = add_death_times(all_history_dict,lhs)
    absolute_tree = dict_to_tree(all_history_dict)
    absolute_tree.show(attr_list=["t_start","t_death"])
    print(history_metrics(absolute_tree))


    print("APPARENT TREE")
    alive_history_dict = parse_histories(alive_language_histories) 
    alive_history_dict = add_death_times(alive_history_dict,lhs)    
    adjusted_history_dict = adjust_horizon(alive_history_dict, 0)
    adjusted_tree = dict_to_tree(adjusted_history_dict)
    adjusted_tree.show(attr_list=["t_start","t_death"])
    print(history_metrics(adjusted_tree))




    #Write to outputfile
    output_row = [os.environ["PRINT_PREAMBLE"]]

    #To measure arratent count
    """for h in np.arange(0,101,10):
        adjusted_history_dict = adjust_horizon(alive_history_dict, h)
        adjusted_tree = dict_to_tree(adjusted_history_dict)  
        output_row.append(history_metrics(adjusted_tree)[2])"""
    
    #To Measure absouate count at time

    

    assert False

    #dict_to_tree(all_history_dict).show(attr_list=["t_start","t_death"])
    for time in np.arange(0 ,int(os.environ["MAX_TIME_STEPS"])+1, 10):
        count = 0
        for l,t in all_history_dict.items():
            if int(t['t_start']) <= time and int(t['t_death']) > time:
                #print(f"Lg {l} alive at time {time} becuase {t}")
                count += 1 
            else:
                #print(f"Lg {l} dead at time {time} becuase {t}")
                pass
        
        

        
        output_row.append(count)

    


    
    with open(os.path.abspath(output_file), 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(output_row)

def add_death_times(hd, lhs):
    #for k,v in hd.items(): print(f"{k} : {v}")
    #print('------------')


    for lh in lhs:
        if lh.split("+")[-2].startswith('DEATH'):
            lang = lh.split("+")[-3].split('_')[-1]
            for k,v in hd.items():
                if k.endswith(f"{lang}/"):
                    hd[k]['t_death'] = lh.split("+")[-2].split('_')[1]
                    #print(hd[k])
    

    #if no death, make it 1e5
    for k,v in hd.items():
        if 't_death' not in v.keys():
            v['t_death'] = '10000'
        else:
            if v['t_death'] is None:
                v['t_death'] = '10000'

    #If splits, consider death of mother lang
    #for k,v in hd.items(): print(f"{k} : {v}")
    #print('------------')
    tree = dict_to_tree(hd)

    for d in tree.descendants:
        if d['t_death'] is None:
            if len(d.children) > 0:
                #print(d.children[0].get_attr('t_start'))
                hd[d.path_name[1:]]['t_death'] = d.children[0].get_attr('t_start')

    #print(hd)

    
    
    #for k,v in hd.items(): print(f"{k} : {v}")
    #print('------------')

    
    
   

    return hd

def sep_alive_langs(lhs):
    all = []
    alive = []

    for lh in lhs:
        if bool(re.search("DEATH_\\d+\+",lh)):
            all.append(re.sub("DEATH_\\d+\+","",lh))
        else:
            alive.append(re.sub("DEATH_\\d+\+","",lh))
            all.append(re.sub("DEATH_\\d+\+","",lh))

    return all, alive


def expand_histories(lhs):

    #for lh in lhs: print(lh)

    #print('-----')

    
    new_lhs = []
    for lh in lhs: new_lhs.append(lh)

    for lh in lhs:
        """pre = '+'.join(lh.split('+')[:-2])
        if pre not in new_lhs: new_lhs.append(pre)
        if len([p for p in pre if len(p)>0]) > 0:
            new_lhs = list(set(new_lhs + expand_histories([pre])))"""
        split_lh = lh.split('+')[:-1]
        #print(split_lh)
        #print('gives:')
        for i,lh_chunk in enumerate(split_lh):
            new_lhs.append('+'.join(split_lh[:i])) 

    

    #for lh in sorted(new_lhs): print(lh)
    #assert False
    return list(set(new_lhs))

def parse_histories(lhs):
        #for lh in sorted(lhs): print(lh)
        #print('==========')

        lhs = expand_histories(lhs)

        #print('==========')
        #for lh in sorted(lhs): print(lh)

        

        ts = []
        tree_hists = []
        for lh in sorted(lhs):
            #print(lh)
            if lh != '':
                tree_string = "INITIAL/"+'/'.join([c[-4:] for c in lh.split("+")])
                tree_hists.append(tree_string)
                ts.append(lh[:-1].split("_")[-2])
               
        cdk_dict = {th:{'t_start':time} for th,time in zip(tree_hists,ts)}

        #print(cdk_dict)

        return cdk_dict

def adjust_horizon(cdk_dict, adjust_time):
    

    
    for v in cdk_dict.values():
        if int(v['t_death']) <= adjust_time:
            v['seen'] = '0'
        else:
            v['seen'] = '1'

    
    new_dict = {k:v for k,v in cdk_dict.items() if bool(int(v['seen']))}
    
    new_dict = squish_dict(new_dict)
    
    return new_dict

    #print(f"Adjustment Time: {adjust_time}")
    """for v in cdk_dict.values():
        #print(f"Old T-start: {v['t_start']}")
        v['t_start'] = str(int(v['t_start']) - adjust_time)
        v['t_death'] = str(int(v['t_death']) - adjust_time)

    
    new_dict = {}
    for k,v in cdk_dict.items():
        if int(v['t_start']) >= 0 and int(v['t_death']) > 0:
            new_dict[k] = v
        else:
            new_dict['INITIAL/'+k.split('/')[-2]] = v

    return new_dict"""


def squish_dict(d):
    #print(d)

    seen_langs = ['INITIAL']
    for k in d.keys():
        if k.endswith('/'):
            seen_langs.append(k.split('/')[-2])
        else:
            seen_langs.append((k+'/').split('/')[-2])

    
    #print(seen_langs)

    
    new_keys = []
    for k in d.keys():
        #print(k)
        if k.endswith('/'):
            new_keys.append('/'.join([l for l in k.split('/')[:-1] if l in seen_langs]))
        else:
            new_keys.append('/'.join([l for l in k.split('/') if l in seen_langs]))

    #new_keys  = ['/'.join([l for l in k.split('/')[:-1] if l in seen_langs])+'/' for k in d.keys()]
    
    #for k,v in d.items():
    #    print(f"{k}\t{v}")

    #print(new_keys)



    new_key_map = {}
    for old in d.keys():
        if old.endswith('/'):
            #print(old[-5:-1])
            new = [nk for nk in new_keys if nk.endswith(old[-5:-1])][0]
        else:
            #print(old[-5:])
            new = [nk for nk in new_keys if nk.endswith(old[-5:])][0]

        #print(f"Replacing {old} with {new}")

        new_key_map[old] = new
    

    new_d = {}
    for k,v in d.items():
        new_d[new_key_map[k]] = v
        
    return new_d

def history_metrics(tree):   
    
    num_lfs = len(tree.children)
    lf_sizes = [len([d for d in lf.descendants])+1 for lf in tree.children]

    return num_lfs, lf_sizes, sum(lf_sizes)




   

    


if __name__ == "__main__":
    
    test_histories = [
            'START_0_L001+', 
            'START_0_L002+', 
            'START_0_L003+START_30_L011+START_54_L015+START_75_L019+', 
            'START_0_L004+START_11_L007+DEATH_69+', 
            'START_0_L005+START_28_L009+DEATH_99+', 
            'START_0_L004+START_11_L006+START_55_L017+', 
            'START_0_L005+START_28_L008+', 
            'START_0_L003+START_30_L010+START_32_L013+', 
            'START_0_L003+START_30_L010+START_32_L012+', 
            'START_0_L003+START_30_L011+START_54_L014+', 
            'START_0_L004+START_11_L006+START_55_L016+START_93_L021+', 
            'START_0_L003+START_30_L011+START_54_L015+START_75_L018+', 
            'START_0_L004+START_11_L006+START_55_L016+START_93_L020+'
        ]
    
    test_histories2 = [
        'START_0_L001+',
'START_0_L002+START_59_L011+',
'START_0_L003+START_14_L007+DEATH_22+',
'START_0_L004+',
'START_0_L005+START_51_L009+',
'START_0_L003+START_14_L006+',
'START_0_L005+START_51_L008+START_62_L013+',
'START_0_L002+START_59_L010+',
'START_0_L005+START_51_L008+START_62_L012+',
]
    test_histories3 = [
'START_0_L001+',
'START_0_L002+START_0_L006+DEATH_51+',
'START_0_L002+START_0_L007+START_43_L016+DEATH_77+',
'START_0_L002+START_0_L007+START_43_L017+',
'START_0_L003+START_4_L008+',
'START_0_L003+START_4_L009+START_37_L014+DEATH_96+',
'START_0_L003+START_4_L009+START_37_L015+START_57_L020+START_58_L022+',
'START_0_L003+START_4_L009+START_37_L015+START_57_L020+START_58_L023+',
'START_0_L003+START_4_L009+START_37_L015+START_57_L021+',
'START_0_L004+START_16_L012+',
'START_0_L004+START_16_L013+',
'START_0_L005+START_4_L010+START_52_L018+',
'START_0_L005+START_4_L010+START_52_L019+START_92_L024+',
'START_0_L005+START_4_L010+START_52_L019+START_92_L025+',
'START_0_L005+START_4_L011+'
    ]


    test_histories4 = [
        'START_0_L001+START_94_L024+',
        'START_0_L001+START_94_L025+',
        'START_0_L002+START_29_L006+DEATH_90+',
        'START_0_L002+START_29_L007+START_55_L018+START_92_L022+',
        'START_0_L002+START_29_L007+START_55_L018+START_92_L023+',
        'START_0_L002+START_29_L007+START_55_L019+',
        'START_0_L003+START_36_L008+START_37_L010+',
        'START_0_L003+START_36_L008+START_37_L011+START_94_L026+',
        'START_0_L003+START_36_L008+START_37_L011+START_94_L027+',
        'START_0_L003+START_36_L009+START_41_L012+START_60_L020+',
        'START_0_L003+START_36_L009+START_41_L012+START_60_L021+',
        'START_0_L003+START_36_L009+START_41_L013+START_46_L014+',
        'START_0_L003+START_36_L009+START_41_L013+START_46_L015+DEATH_81+',
        'START_0_L004+START_53_L016+',
        'START_0_L004+START_53_L017+DEATH_96+',
        'START_0_L005+'
    ]
    history_analysis(test_histories4, None)
