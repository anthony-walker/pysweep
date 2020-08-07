import os,sys,yaml,errno,traceback

pysweepPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
equationPath = os.path.join(pysweepPath,"equations")
def generateInputFile(name):
    """Use this function to generate predefined input files.
        args: name - this is the prefix of the equation files, e.g., name="euler" for euler.py and euler.cu
    """
    equationFiles = os.listdir(equationPath)
    equationFiles.remove("__init__.py")
    try:
        pycacheDir = "__pycache__"
        if pycacheDir in equationFiles:
            equationFiles.remove(pycacheDir)
        if name+".py" not in equationFiles:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name+".py")
        if name+".cu" not in equationFiles:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name+".cu")
    except Exception as e:
        tb = traceback.format_exc()
        tb+= "\nOptions:\n{}".format(equationFiles)
        print(tb)
    
    yamlOutput = {"swept":True,"filename":"{}.hdf5".format(name),"verbose":True,"blocksize":8,"share":0.5,"dtype":"float64","globals":["FIXME",],"intermediate_steps":2,"operating_points":2,"cpu":"","gpu":""}

    for file in equationFiles:
        if name in file:
            if ".py" in file:
                yamlOutput["cpu"] = os.path.join(equationPath,file)
            if ".cu" in file:
                yamlOutput["gpu"] = os.path.join(equationPath,file)
    with open("{}.yaml".format(name),"w") as f:
        yaml.dump(yamlOutput,f)