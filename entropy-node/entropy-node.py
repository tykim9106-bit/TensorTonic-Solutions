import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    
    map = {}
    for class_ in y:
        map[class_]=map.get(class_, 0) + 1

    z = np.array(list(map.values()))
    zz = z/z.sum()
    zzz = np.where(zz>0, zz * np.log2(zz), 0)
    return -zzz.sum()