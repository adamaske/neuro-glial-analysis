
def li_f1(a, b):
    li = (a - b) / (a + b)
    return li

def li_f2(a,b):
    li = (a - b) / (abs(a) + abs(b))
    return li

def li_f3(a,b):
    li = (abs(a) - abs(b)) / (abs(a) + abs(b))
    return li

def test_vals(a, b):
    print(f" LI : {a} vs {b}")
    f1 = li_f1(a, b)
    f2 = li_f2(a, b)
    f3 = li_f3(a, b)

    print("f1 : ", f1) 
    print("f2 : ", f2) 
    print("f3 : ", f3) 
    
    return f1, f2, f3


test_vals(1.2, 0.1)
test_vals(1.2, -0.1)
test_vals(-1.2, 0.1)
test_vals(-1.2, -0.1)

