
def label_numeric_to_number(label)->int:
    """Convert numeric label to number(integer)"""
    if label == "zero":
        return 0    
    elif label == "one":
        return 1
    elif label == "two":
        return 2
    elif label == "three":
        return 3
    elif label == "four":
        return 4
    elif label == "five":
        return 5
    elif label == "six":
        return 6
    elif label == "seven":
        return 7
    elif label == "eight":
        return 8
    elif label == "nine":
        return 9


def sum_(num1:int,num2:int):
    return num1+num2
def substract(num1:int,num2:int):
    return num1-num2
def div(num1:int,num2:int):
    return num1/float(num2)
def times(num1:int,num2:int):
    return num1*num2

def label_symbolic_to_expression(label):
    """Return function of expression"""
    if label == "plus":
        return sum_
    elif label == "minus":
        return substract
    elif label == "div":
        return div
    elif label == "decimal" or label == "times":
        return times

def label_expression_to_str(label):
    """Return symbol of expression"""
    if label == "plus":
        return '+'
    elif label == "minus":
        return '-'
    elif label == "div":
        return '/'
    elif label == "decimal" or label == "times":
        return 'x'


def solve(num_1:str,num_2:str,expression:str):
    num_1_ = label_numeric_to_number(num_1)
    num_2_ = label_numeric_to_number(num_2)
    if num_2_ == 0 and expression=='div':
        return None
    if num_1_ is None or num_2_ is None:
        return None 
    fnc = label_symbolic_to_expression(expression)
    return str(fnc(num_1_, num_2_))


# Test function for all cases
def test_case(num1_str,num2_str,num1_,num2_):
    assert( float(solve(num1_str,num2_str,"times"))      == times(num1_,num2_))
    assert( float(solve(num1_str,num2_str,"decimal"))    == times(num1_,num2_))
    assert( float(solve(num1_str,num2_str,"plus"))       == sum_(num1_,num2_))
    assert( float(solve(num1_str,num2_str,"minus"))      == substract(num1_,num2_))
    if num2_ != 0:
        assert( float(solve(num1_str,num2_str,"div"))        == div(num1_,num2_))
    else:
        assert( solve(num1_str,num2_str,"div")        == None)

def test():
    # Create list of numbers and their string counterparts,
    # Then select random couples from this list to test randomly!
    import random
    nums_list = [
        ("zero",0),
        ("one",1),
        ("two",2),
        ("three",3),
        ("four",4),
        ("five",5),
        ("six",6),
        ("seven",7),
        ("eight",8),
        ("nine",9)]
    num_test_cases = 500
    for case in range(num_test_cases):
        num_1_list = random.choice(nums_list)
        num_2_list = random.choice(nums_list)
        test_case(num_1_list[0],num_2_list[0],num_1_list[1],num_2_list[1])

if __name__ == '__main__':
    test()