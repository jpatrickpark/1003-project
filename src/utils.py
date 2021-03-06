def dotProduct(d1, d2):
    """
    @param list d1: a dict. a sparse representation of a row. 
    @param list d2: a dict. a sparse representation of a row.
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())
    
def ListToDict(l1):
    if type(l1)==list:
        return dict((x, 1) for x in l1)
    else:
        raise TypeError("This is not a list.")

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = [key] 
            else: 
                inverse[item].append(key) 
    return inverse