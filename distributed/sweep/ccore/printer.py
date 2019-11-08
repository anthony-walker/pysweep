from collections import Iterable
import sys

class pysweep_printer(object):
    """This function will store the master rank and print given values."""

    def __init__(self, rank,master_rank):
        self.rank = rank
        self.master = master_rank

    def __call__(self, args,p_iter=False,p_ranks=False,end="\n"):

        if (self.rank == self.master or p_ranks) and p_iter:
            if isinstance(args,Iterable):
                for item in args:
                    sys.stdout.write("[ ")
                    for si in item:
                        sys.stdout.write("%.0f"%si+", ")
                    sys.stdout.write("]\n")
            else:
                args = args,
                for item in args:
                    print(item,end=end)
        elif self.rank == self.master or p_ranks:
            print(args,end=end)

def pm(arr,i,iv=0,ps="%d"):
    for item in arr[i,iv,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write(ps%si+", ")
        sys.stdout.write("]\n")
