class sweep_lambda(object):
    """This class is a function wrapper to create kind of a pickl-able lambda function."""
    def __init__(self,args):
        self.args = args

    def __call__(self,x):
        sweep_fcn,cpu_fcn,ts,ops = self.args
        return sweep_fcn(x,cpu_fcn,ts,ops)
