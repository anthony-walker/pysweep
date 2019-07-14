class sweep_lambda(object):
    """This class is a function wrapper to create kind of a pickl-able lambda function."""
    def __init__(self,args):
        self.args = args

    def __call__(self,block):
        sweep_fcn,source_mod,ops = self.args
        return sweep_fcn((block,source_mod,ops))
