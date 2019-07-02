# Programmer: Anthony Walker
"""The purpose of this file is post processing of data obtain
from the fluid domain."""
#READ ME:
#Use this to plot all of the data in a folder that is created by the
#euler1D code.
import os
import sys
import matplotlib as mpl
import difflib as dl
mpl.use("Tkagg")
import matplotlib.pyplot as plt
import re
import numpy as np
import imageio

class ATK(object):  # A function to combine all methods
    """Use this class to create an analysis tool kit"""
    def __init__(self):
        #Instance Variables
        self.mD = os.getcwd()+"/solutionData/" #main directory
        self.gPath = self.mD +"/GIFs/" #GifPath
        self.cd = None #current working directory
        self.imlist = list()
        self.initialize() #Initializes other instance variables
        #Functions to create instance Variables
        self.mkCmdList()
        self.run()

#USERCOMMANDS
    def chdir(self,GA = True):
        dirList = os.listdir(self.mD)
        print("Avaliable Directories:")
        print("*----------------*")
        for x in dirList:
            print(x)
        print("*----------------*")
        if GA:
            cd = input("Please enter the directory you wish to analyze. ")
        else:
            cd = input("The directory entered is invalid. Please enter a new directory. ")
        if cd in dirList:
            self.cd = os.getcwd()+"/solutionData/"+cd
            for i in os.listdir(self.cd):
                self.files[i] = None
        else:
            self.chdir(GA = False)

    def cmds(self):
        """Use this function to list commands."""
        print("Avaliable Commands:")
        print("*----------------*")
        for c in self.cmdsList:
            print(c)
        print("*----------------*")

    def dataPlot(self, fN = None, index = None, gDirec = None):
        """Use this method to plot data."""
        if fN is None:
            self.rDF()
        else:
            self.rDF(fileName = fN)

        if index is None:
            index = self.keyRequest()

        self.makePositionData(self.cFI)
        data = self.data[self.cFI]
        y = tuple()
        for d in data:
            y+=(d[index],)
        x = self.ranges[self.cFI][0]
        fig = plt.figure(1)
        plt.plot(x,y)
        if fN is None:
            plt.show()
            print("waiting for plot to close...")
        else:
            gpc = self.gPath+gDirec+"/images"
            gn = 0
            gf = "gif"+str(gn)+".png"
            while gf in os.listdir(gpc):
                gn+=1
                gf = "gif"+str(gn)+".png"
            plt.savefig(gpc+"/"+gf)
            self.imlist.append(gpc+"/"+gf)
            plt.gcf().clear()

    def dataContour2D(self, fN = None, index = None, gDirec = None):
        """Use this method to plot data contours."""
        if fN is None:
            self.rDF()
        else:
            self.rDF(fileName = fN)

        if index is None:
            index = self.keyRequest()

        self.makePositionData(self.cFI)
        data = self.data[self.cFI]
        plotIndicies = self.makeIndices(tuple(),self.dims[self.cFI])
        #Contour Data
        print(self.dims)
        x = self.ranges[self.cFI][0]
        y = self.ranges[self.cFI][1]
        z = np.zeros(self.dims[self.cFI])
        for i in range(len(plotIndicies)):
            z[plotIndicies[i]]+=data[i][index]
        #Plotting
        fig = plt.figure(1)

        #Add Plot Formating Here
        plt.contourf(x,y,z)
        if fN is None:
            plt.show()
            print("waiting for plot to close...")
        else:
            gpc = self.gPath+gDirec+"/images"
            gn = 0
            gf = "gif"+str(gn)+".png"
            while gf in os.listdir(gpc):
                gn+=1
                gf = "gif"+str(gn)+".png"
            plt.savefig(gpc+"/"+gf)
            self.imlist.append(gpc+"/"+gf)
            plt.gcf().clear()

    def help(self):
        """Use this function to print help."""
        with open('supportingFiles/ATKRM.txt','r') as f:
            line = f.read()
            while line:
                print(line)
                line = f.read()
        f.closed

    def clear(self):
        """Use this to clear the screen."""
        os.system("clear")

    def exit(self):
        """Use this function to stop execution."""
        pass

    def makeGIF(self):
        """Use this function to create a gif."""
        print("GIF creation initiated.")
        self.imlist = list()
        self.chdir() #Creating directory
        if not os.path.isdir(self.gPath):
            os.mkdir(self.gPath)
        #Making gif directory
        gDir = input("Please name the directory to store your gif: ") #gif directory
        while os.path.isdir(self.gPath+gDir):
            gDir = input("Directory already exists, please try again: ")
        os.mkdir(self.gPath+gDir)
        os.mkdir(self.gPath+gDir+"/images")
        #Getting and sorting directory files
        dFL = os.listdir(self.cd)
        ndFL = list()
        i = 0
        for fn in dFL:
            for num in re.findall(r'_\d+_',fn):
                ndFL.append([int(num[1:len(num)-1]),i])
                i+=1
        sFL = sorted(ndFL,key = lambda x: x[0])
        sortedFiles = list()
        for item in sFL:
            sortedFiles.append(dFL[item[1]])
        #requesting plotting function
        self.cmds()
        pf = input("Please enter the plotting function: ") #gif directory
        while pf not in self.cmdsList:
            self.cmds()
            pf = input(pf+": Command not found, please try again: ")
        #key request
        self.rDF(fileName = sortedFiles[0])
        idx = self.keyRequest()
        #Creating gif images
        print("Creating GIF image files...")
        for cf in sortedFiles:
            exeStr = "self."+pf+"("+"fN=cf,"+"index=idx,"+"gDirec=gDir)"
            exec(exeStr)
        print("Writing images to GIF file...")
        with imageio.get_writer(self.gPath+gDir+"/"+gDir+'.gif', mode='I') as gifWriter:
            for im in self.imlist:
                image = imageio.imread(im)
                gifWriter.append_data(image)
        print("GIF creation complete.")
#COMMANDSEND

    ####Other Plot Functions

    def makePositionData(self,index):
        """Use this to create positions via ranges."""
        r = self.steps[index]
        print(r)
        rT = tuple()
        dims = tuple()
        i = 0
        for s in r:s
            rT += (np.arange(s[0],s[1]+s[2],s[2]),)
            dims +=(len(rT[i]),)
            i+=1
        self.ranges.append(rT)
        print(dims)
        self.dims.append(dims)

    def makeIndices(self,rData,dims,point=None,index=0,getIndex = None):
        """Use this method to create all points based on dimensions"""
        if point is None:
            point = np.zeros(len(dims))
        if index < len(dims):
            for i in range(dims[index]):
                rData = self.makeIndices(rData,dims,point,index+1,getIndex)
                if point[index] == dims[index]-1:
                    point[index] = 0
                else:
                    point[index] += 1
            return rData
        else:
            pT = tuple()
            for x in point:
                pT+=(int(x),)
            if getIndex is not None:
                pT+=(getIndex,)
            rData += (pT,)
            return rData

    #### ATK Run Functions
    def run(self):
        inval = "chdir"
        print("Welcome to the Analysis toolkit.")
        print("Here you can enter commands to analyze data or type exit to exit.")
        print("If you are unsure of avaliable commands you can type \'cmds\'.")
        print("you can also type. \'help\' for more detailed information.")
        print("To get started, you must enter a directory.")
        while inval != 'exit':
            inval = self.exeCmd(inval)
            if inval != 'exit':
                inval = input(":")

    def exeCmd(self,strg):
        """This command executes the input string."""
        try:
            if strg in self.cmdsList:
                exec("self."+strg+"()")
            else:
                eStr = strg+": Command not found, type \"cmds\" for a list of commands."
                raise(Exception(eStr))
        except Exception as e:
            print(e)

    def mkCmdList(self):
        """Use this method to create the list of commands."""
        sB = False
        with open(__file__,'r') as atkF:
            for line in atkF:
                if line.__contains__('USERCOMMANDS'):
                    sB = True
                elif line.__contains__('COMMANDSEND'):
                    sB = False
                if sB:
                    if line.__contains__("def "):
                        strs = line.split("(")
                        self.cmdsList.append(strs[0][8:])
        atkF.closed

    def keyRequest(self,GA = True):
        """Use this method to print avaliable data keys."""
        print("Avaliable Data:")
        print("*----------------*")
        for key in self.dataKeys:
            print(key)
        print("*----------------*")

        if GA:
            pk = input("Please enter the data key you would like to plot: ")
        else:
            pk = input("Invalid key, please enter a new data key: ")

        if pk in self.dataKeys:
            return self.dataKeys[pk]
        else:
            return self.keyRequest(GA = False)

    ####File Reading Functions
    def rDF(self,GA=True,fileName = None):
        "Use this method to read a domain file"
        if GA and fileName is None:
            print("Avaliable Files:")
            print("*----------------*")
            for c in self.files:
                print(c)
            print("*----------------*")
            fileName = input("Please enter a file to be plotted: ")
        elif fileName is None:
            fileName = input("Invalid filename, please enter a new file to be plotted: ")

        if fileName in self.files:
            if self.files[fileName] is not None:
                self.cFI = self.files[fileName]
            else:
                self.files[fileName] = self.next
                self.cFI = self.files[fileName]
                self.next+=1
                points =  tuple()
                datasets = tuple()
                dStrs = tuple() #data strings
                pStrs = tuple() #point strings
                #reading file
                with open(self.cd+"/"+fileName,'r') as f:
                    self.headingList(f)
                    for line in f:
                        temp = line.split(":")
                        pStrs+=(temp[0],)
                        dStrs+=(temp[1],)
                f.closed
                #making points
                for p in pStrs:
                    point = tuple()
                    for intnum in re.findall(r'\d+',p):
                        point+=(int(intnum),)
                    points+=(point,)
                #making data
                for d in dStrs:
                    data = tuple()
                    for flonum in re.findall(r'-?\d+.\d+',d):
                        data+=(float(flonum),)
                    datasets+=(np.array(data),)
                self.data.append(datasets)
                self.points.append(points)
        else:
            self.rDF(GA = False)

    def headingList(self,f):
        """Use this method to get heading information."""
        tta = tuple()
        line = f.readline()
        line = line[0:len(line)-1]
        tta +=(line,)
        for i in range(2):
            line = f.readline()
            for intnum in re.findall(r'\d+',line):
                tta+=(int(intnum),)
        rT = tuple()
        for i in range(2):
            line = f.readline()
            for flonum in re.findall(r'-?\d+.\d+',line):
                rT+=(float(flonum),)
        self.headings.append(tta)
        self.steps.append(self.stepReorganize(rT))
        line = f.readline()
        line = line[0:len(line)-1]
        line = "self.dataKeys =" + line
        exec(line)
        f.readline() #skip data separation line

    def stepReorganize(self,data):
        """Use this method to organize step data."""
        r = tuple()
        r+=((data[0],data[1],data[-1],),)
        for i in range(1,int(len(data)/3)):
            r += ((data[i+1],data[i+2],data[-1*(i+1)],),)
        return r

    ####Other Functions
    def initialize(self):
        """Use this function to initialize data for new directories."""
        self.headings = list() #Heading information from files
        self.steps = list() #stepsize and range information
        self.data = list() #data from the file
        self.points = list() #created points for the data
        self.ranges = list() #created ranges for the data
        self.dims = list() #dimensions of the data
        self.cmdsList = list() #commands for the ATK
        self.files = dict() #current file list in cd
        self.next = 0 #Next file index to add
        self.cFI = None #Current File Index
        self.dataKeys = None #current indexs for data in files
#Main function
if __name__ == "__main__":
    atk = ATK()
