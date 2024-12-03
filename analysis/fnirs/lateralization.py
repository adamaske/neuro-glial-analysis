

def lateralization_index(x, y):
    if (x + y) == 0: #divide by zero 
        return 0
    
    li = (x - y) / (x + y)

    return li