

def history_figure(lhs):
    coord_dict = {}
    for lh in lhs:
        lh_split = lh.split('_')
        coord_dict.update({lh_split[-2] : (None,lh_split[-3])})

    
    print(coord_dict)

    num_heads = len([1 for c in coord_dict.values() if c[1] == '0'])










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