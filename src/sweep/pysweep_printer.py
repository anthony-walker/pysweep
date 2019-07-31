from collections import Iterable

class pysweep_printer(object):
    """This function will store the master rank and print given values."""

    def __init__(self, rank,master_rank):
        self.rank = rank
        self.master = master_rank

    def __call__(self, args,p_iter=False,p_ranks=False):

        if (self.rank == self.master or p_ranks) and p_iter:
            if isinstance(args,Iterable):
                for item in args:
                    print(item)
            else:
                args = args,
                for item in args:
                    print(item)
        elif self.rank == self.master or p_ranks:
            print(args)
