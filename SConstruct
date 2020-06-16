# @Author: Anthony Walker <sokato>
# @Date:   2020-04-07T13:57:13-07:00
# @Email:  walkanth@oregonstate.edu
# @Filename: SConstruct
# @Last modified by:   sokato
# @Last modified time: 2020-04-16T15:01:17-07:00



import sys, os, subprocess, traceback
#Finding nvcc
try:
    process = subprocess.Popen(['which', 'nvcc'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    nvcc_path, err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError("nvcc not found. Please make sure it can be found with \'which nvcc\'.")
    nvcc_path = os.path.dirname(nvcc_path.decode("utf-8") )
except Exception as e:
    tb = traceback.format_exc()
    print(tb)
    sys.exit()
#Update nvcc path
env = Environment()
env.Tool('cuda')
# env.Append(LIBS=['cutil', 'glut', 'GLEW'])

# env['CUDA_TOOLKIT_PATH'] = path to the CUDA Toolkit
# env['CUDA_SDK_PATH'] = path to the CUDA SDK
#
# env['NVCCFLAGS'] = flags common to static and shared objects
# env['STATICNVCCFLAGS'] = flags for static objects
# env['SHAREDNVCCFLAGS'] = flags for shared objects
