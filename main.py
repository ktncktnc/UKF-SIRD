import sys
from .model import SIRD

if if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print("Err: Wrong parameter")
        print("Inf: main.py <Country> <Number_of_Days>")
        return

    model = SIRD(args[1])
    model.solve(int(args[2]))
    model.plot()