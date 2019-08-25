
import inspect

def get_var_name(var):
    """Use this function to retrieve variable name"""
    vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in vars if var_val is var]

def other_fcn(args):
    A = 10
    print(get_var_name(args[0]))

A = 10
args = (A,)
other_fcn(args)
