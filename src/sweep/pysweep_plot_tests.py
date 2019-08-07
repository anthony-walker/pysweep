#Programmer: Anthony Walker
#This file contains test created during the creation of sweep
#It is strictly for debugging
import matplotlib as mpl
mpl.use("Tkagg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import imageio
import os


def plot_point_set():
    """Use this function to plot a set of points"""
    pass

def plot_swept_step(pts,name="./paper/figures/pyramid.gif",gf = "./paper/figures/pyramid.png"):
    """Use this function to plot the evaluated points of a swept step."""
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection='3d') #Creating 3D axis
    ax.view_init(elev=45., azim=45)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xlim([0,256])
    ax.set_ylim([0,256])
    ax.set_zlim([0,15])
    ax.set_zlabel(' Time')
    colors = ['b','r','g','c','y','k','m','r']
    cl = len(colors)
    sch = 4
    sl= 64
    with imageio.get_writer(name, mode='I',fps=10) as gifWriter:
        color_ctr = 0
        for i in range(len(pts[0])):
            sx = 0
            sy = 0
            for ptl in pts:
                if color_ctr == cl:
                    color_ctr = 0
                cc = colors[color_ctr]
                color_ctr+=1
                X = [x[0]+sl*sx for x in ptl[i]]
                Y = [x[1]+sl*sy for x in ptl[i]]
                Z = [i for x in ptl[i]]
                plt.plot(X,Y,Z,linestyle=None,marker='o',color=cc)
                sx += 1
                if sx == sch:
                    sy += 1
                    sx = 0
            fig.savefig(gf)
            image = imageio.imread(gf)
            gifWriter.append_data(image)
        # os.remove(gf)
    print("Finished")
