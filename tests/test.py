# test module

def printname(x):
    print __name__

def fun1(x):
    print x

if __name__ == "__main__":
    import sys
    printname(int(sys.argv[1]))
