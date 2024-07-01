###################################
##importing essential libraries
###################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import pandas as pd
import math
import datetime

########################################
##Path to files
#######################################
data_dir = '/home/yogender/Desktop/Readings/6010New200TemPlots+Rainbow/'
#filename = '2.txt'
litersPerHour = ['800', '600', '400','200']
filepaths = [data_dir + 'ALMEMO' + x + '.001'for x in litersPerHour]
colors = ['m', 'g', 'r', 'k']
listMarkers= ["+","o","v","^","<",">","s","p","*","h","x","D","d","1","2","3","4","d","|","X"]
Temlegends = ["1","2","3","4","5","6","7","8", "9","10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
reversedTemlegends = Temlegends[::-1]
#TemProfileLitersPerHour = ['800l/h', '600l/h', '200l/h']
TemProfileLitersPerHour = [x +"l/h" for x in litersPerHour]

temProfileMarkers = [listMarkers for i in range(0,len(litersPerHour))]
print (len(temProfileMarkers))
print (type(temProfileMarkers))
for x in temProfileMarkers:
    print (x)
    print (len(x))
    print (type(x)) 
#listColors = pd.DataFrame(["b","g","r","c","m","y","k","b","g","r","c","m","y","k","b","g","r","c","m","y"])
listColors = ["b","g","r","c","m","y","k","b","g","r","c","m","y","k","b","g","r","c","m","y"]
temProfileColors = [listColors for i in range (0,len(litersPerHour))]
print (len(temProfileColors))
print (type(temProfileColors))
for x in temProfileColors:
    print (x)
    print (len(x))
    print (type(x))
temProfileMarkers = [listMarkers for i in range(0,3)]
print (len(temProfileMarkers))
print (type(temProfileMarkers))
for x in temProfileMarkers:
    print (x)
    print (len(x))
    print (type(x)) 
##temProfileColors = pd.DataFrame([["b","g","r","c","m","y","k","b","g","r","c","m","y","k","b","g","r","c","m","y"],
##                    ["b","g","r","c","m","y","k","b","g","r","c","m","y","k","b","g","r","c","m","y"],
##                    ["b","g","r","c","m","y","k","b","g","r","c","m","y","k","b","g","r","c","m","y"]])



pointtypes = ['-.', '-.', '-.', '-.']
markers = ['+','x', 'o', 'v']
labels = ['800l/h', '600l/h', '400l/h', '200l/h']
out_dir = '/home/yogender/Desktop/Readings/6010New200TemPlots+Rainbow/plots/'
out_dirs = [out_dir + x + "l_h" for x in litersPerHour]

os.makedirs(out_dir, exist_ok = True)
##############################################
##Function to create the df with different skiprows as input, depending upon the files produced by ALMEMO
##I can now create more files from ALMEMO and read them all together.
#############################################

def read_data(fpath):
    if fpath == filepaths[0]:
        df = pd.read_csv(os.path.join(fpath), sep = ";", skiprows = 21, header = None)
    elif fpath == filepaths[1]:
        df = pd.read_csv(os.path.join(fpath), sep = ";", skiprows = 21, header = None)
    elif fpath == filepaths[2]:
        df = pd.read_csv(os.path.join(fpath), sep = ";", skiprows = 25, header = None)
    elif fpath == filepaths[3]:
        df = pd.read_csv(os.path.join(fpath), sep = ";", skiprows = 30, header = None)
        
    return df
############################################
##Function to read filepaths as a list. And store them in a list. So its a list of pandas dataframe.
##Noteworthy point is that every dataframe is stored in a list so I have to enter a list to perform pandas functions. 
############################################
def path2df(paths):        
        dfs = [read_data(p) for p in paths]
        return dfs
    
Dfs = path2df(filepaths)
print (Dfs)

###############################################
##Function to read the temperature values to analyze. I is the list of range of coloums to slice from data.
###I is using list() whic converts the iteratable sych as tuples, sets dictionaries to list.iloc is pandas slicing. So I have
#### looped inside the list to enter the pandas dataframe to perform iloc function.
##############################################
def data2Code(dfs):
        I = list(range(62, 72)) + list(range(82, 92))
        dataframe = [df.iloc[:, I] for df in dfs]
        return dataframe
Tem2Code = data2Code(Dfs)
print ('Tem values to code upone = ', Tem2Code)
print (type(Tem2Code))

############################################
##Function to retrieve the date string from DFs
############################################

def dateString_l_of_dfs(dfs):
    #I = list(range(20,300))
    #I = list(range(1140,1476))
    datestring = [df.iloc[:,1]for df in dfs]
    return datestring
dates_list_of_Dfs = dateString_l_of_dfs(Dfs)
print ('date =',  dates_list_of_Dfs)
#print (type(Dfs))
print (np.shape(dates_list_of_Dfs))
print (len(dates_list_of_Dfs))
print (type(dates_list_of_Dfs))

#################################
##Function to parse date string as datetime object by (datetime.datetime.strptime(x, '%H:%M:%S'))
##and then by performing (.time ) in end
#################################
def datestring_to_datetime(datestr):
    val = [[datetime.datetime.strptime(x, '%H:%M:%S').time() for x in y ]for y in datestr]    
    return val
dateString2dateTime = datestring_to_datetime(dates_list_of_Dfs)

################################
##Function to perfrom nondimensionalisation of time. Note that I have used pd.to_datime to convert the date time series in
##dateString2dateTime to pandas date time series so I could easily perform subtraction and division of timestamp and timedelta object
################################

def scale_it_pseries(pseries):
    timString2PdTime=[pd.to_datetime(s,format='%H:%M:%S') for s in pseries]
    denominator = [p.max() - p.min() for p in timString2PdTime]
    nominator = [p - p.min() for p in timString2PdTime]
    scaled_value = [(n/d) for n,d  in zip (nominator,denominator)]
    return scaled_value
timeScaled = scale_it_pseries(dateString2dateTime)
print (timeScaled)


######################################
##Function to replace commas with dot as it will later be converted to float. with comma it cannot be converted to float. 
##I dont know how df.stack and df.unstack works, but the matter of fact is it worked. Google for more info.
######################################
def replace_comma(dfs):
    val = [df.stack().str.replace(',', '.').unstack() for df in dfs]
    return val

comma2Dot = replace_comma(Tem2Code)
print (comma2Dot)
print (type(comma2Dot))
###################################
##Conversion of string to float. It is somtimes necessary to convert pandas data to float. Just in case its a good practise
##################################
def floatdata(dfs):
    val = [df.applymap(float) for df in dfs]
    return val
string2float = floatdata(comma2Dot)
print (string2float)
#################################
##taking average of all the temperature to find the mixed temperature at each time step. So this is time averaged temperature.
##Mind that again the df.mean is the pandas function and I have performed it while looping inside the list over the two set of p
##pandas dataframe.
################################
def Tmix_average(dfs):
    val = [df.mean(axis = 1) for df in dfs]
    return val
Tmix = Tmix_average(string2float)
print (Tmix)
##################################
##Moment of energy of mixed tank = rho*cp*V*h. Note that the I have looped over two dataframes in the list and then looped over
##each element of each dataframe.
###################################
def Memix(dfs):
    val = [[1000*4.186*0.397*x*0.835 for x in df] for df in dfs]
    return val
MEmix = Memix(Tmix)
print (MEmix)
print (type(MEmix))
print (np.shape (MEmix))
print (len(MEmix))
for i in MEmix:
    print ('inner list in MEmix =', len(i))
##################################
##Exp is the energy of each layer of fluid element in the tank so its 20 coloumns and val is a list of 2 x 20
##coloums, so 20 per pandas dataframe. Noteworthy point is the return val is behaving as the lambda function in pandas.
##Since I have looped over the dataframes in the list and not over each element in the df. its apparant that the compiler is
##recognising that its a pandas data frame inside the list so taking this formulae along with x equivalent to lambda function
##in pandas and performing it element wise without the need to loop over each element.
##################################
def Exp(dfs):
    val = [1000*4.186*(0.397/20)*x for x in dfs]
    return val
Eexp = Exp(string2float)


#####################################
##Function to create a numpy array of sensors height and finally saving it as node of fluid element.
##doggy shit
####################################
def nodesHeight():
    
    x1 = 0.15
    x2 = 1.75
    HeightOfSensors = np.arange(1.67, 0.07, -0.08)
    #print (HeightOfSensors)

    val = np.array([x-0.04175 for x in HeightOfSensors])
    return val

HeightOfNodes = nodesHeight()
print ('height of nodes = ', HeightOfNodes)

print (type(HeightOfNodes))
##########################################
##Multiplying each element in df with the height of node coloumn wise. Once again see that I have looped over the dfs and not
##the element of dfs. This is done by the .x multiply function of pandas.Super cool. I am dodging over list storing multiple data frames.
########################################
def MEexp(dfs):
    #for i in HeightOfNodes:
        val = [x.multiply(HeightOfNodes, axis=1) for x in dfs]
        return val
Mexp = MEexp(Eexp)

print ('Mexp = ', Mexp)

#################################
##Summing the Mexp coloum wise
#################################
def sum_ME_Exp(dfs):
    val = [x.sum(axis = 1) for x in dfs]
    return val

ME_sum = sum_ME_Exp(Mexp)

print('ME_sum = ', ME_sum)
##################################
##Summing Eexp
##################################
def sum_Eexp(dfs):
    val = [x.sum(axis = 1) for x in dfs ]
    return val

E_sum = sum_Eexp(Eexp)
#print ('E_sum=',E_sum)
#print (type(E_sum))
#print (np.shape(E_sum))
##

#########################################
##Creating a list of A dataframe.df.shape[0] will take the shape of df in dfs, zero is passed as the index which means
##number of rows in df, for df in dfs. this number of rows ar then multiplied by [A]-> [A,A,A,A...n] n being number of rows in df
##########################################
def y_A(dfs):
    th = 60
    tc = 10
    A = (1.67*th)/(th-tc)
    val = [df.shape[0] * [A] for df in dfs]
    return val

A_y = y_A(E_sum)

####for i in A_y:
##    print ('len of horse shit =', len(i))
#print (A_y)
print ('len of A = ', len(A_y))

############################################
##Creating a list of B dataframe, an empty list, appending it to list_dfs. Note that I have looped over df in dfs and then
##each element in df as well to perform the mathmatical calculation and then append it to the list. It can be done with pandas
##lambda function as well
##############################################
                
def y_B(dfs):
    list_dfs = [[] for j in range(0, len(litersPerHour))]
    print ('len of list_dfs = ', len(list_dfs))
    for i, df in enumerate(dfs):
        th = 60
        tc = 10
        for x in df:
            y = x/(1000*4.186*(math.pi)*(0.55**2)*(1/4)*(th-tc))
            list_dfs[i].append(y)
    return list_dfs
        
B_y = y_B(E_sum)
print ('len of bullcrap = ',  len(B_y))
print (B_y)

#print (np.shape(B_y))
#print ('len of bullcrap = ',  len(A_y))
#print (A_y)
#print (np.shape(A_y))
#################################################
##Subtracting A from B. looping over A and B, then each elemnet in innerlist of A and B. Easy Pesy.
################################################
def y(A, B):
    c = [[i-j for i,j in zip (x,y)] for x,y in zip (A, B)]
    #c=[map(lambda x, y: x-y, ii, jj) for ii, jj in zip(A,B)]
    return c
   
Y = y(A_y, B_y)
##df = pd.DataFrame(Y)

print ('Y = ', Y)
for j, i in enumerate(Y):
    print ('length of inner list in Y =', len(i))
    print ('value of inner list i = ',  j, len(i), i )

print ('length of Y =', len(Y))
print ('shape of Y=',np.shape(Y))
print ('shape of B_y = ', np.shape(B_y))
print ('length of B_y = ', len(B_y))

##############################################
##Calculating moment of energy of cold region. looping over the list A and then the inner list in A.
##############################################
def MstrC(A):
    th = 60
    tc = 10
    val = [[(math.pi)*(0.55**2)*(1/4)*i*1000*4.186*th*(i)/2 for i in x] for x in A]
    return val
MstrCold = MstrC(Y)
print (MstrCold)
print ('MstrCold =', len(MstrCold))
print ('MstrCold=',np.shape(MstrCold))

for j, i in enumerate(MstrCold):
    print ('length of inner list in MstrCold =', len(i))
    print ('value of inner list MstrCold= ',  j, len(i), i)

#################################################
##Calculating moment of energy of hot region. looping over the list A and then the inner list in A.
################################################
def MstrH(A):
    th = 60
    tc = 10
    val = [[(0.397 - ((math.pi)*(0.55**2)*i*(1/4)))*1000*4.186*th*(((1.67-i)/2)+i) for i in x] for x in A]
    return val
MstrHot = MstrH(Y)
#print (MstrHot)
print ('MstrHot=', len(MstrHot))
print ('MstrHot=',np.shape(MstrHot))
for j, i in enumerate(MstrHot):
    print ('length of inner list in MstrHot =', len(i))
    print ('value of inner list MstrHot= ',  j, len(i), i)
######################################################
##Taking element wise sum of A and B. looping in list A and B and then in inner list of A and B respectively. 
#####################################################
def Mstr_sum(A,B):
    val = [[i+j for i,j in zip(x,y)] for x, y in zip (A,B)]
    return val
Mstr = Mstr_sum(MstrCold, MstrHot)
#print (Mstr)
print ('Mstr=', len(Mstr))
print ('Mstr=',np.shape(Mstr))
    
for j, i in enumerate(Mstr):
    print ('length of inner list in Mstr =', len(i))
    print ('value of inner list Mstr= ',  j, len(i), i)
#############################################
##Taking element wise subtraction and divion of A, B, C. looping in list A, B, C and then in inner list of A, B, C respectively. 
#############################################
def Mix (A,B,C):
    val = [[(a-b)/(a-c) for a,b,c in zip(x,y,z)] for x,y,z in zip (A,B,C)]
    return val
MixNumber = Mix(Mstr, ME_sum, MEmix )

print (MixNumber)
print ('Mstr=', len(MixNumber))
print ('Mstr=',np.shape(MixNumber))

for i, mix in enumerate (MixNumber):
    print ('innetr list in MixNumber= ', i, len(mix), mix )
print (timeScaled)
print ('nondimensionalised time', len(timeScaled))
print ('nondimensionalised time',np.shape(timeScaled))
print (type(timeScaled))
for x in timeScaled:
        print (type(x))

###########################################
##Matplot function to plot multiple plots on nondimensionalised time
#############################################

def multiplot(mixnumber,
              timedata,
              outpath,
              colors,
              point_types,
              labels,
              markers,
              xlabel = "Dimensionless time",
              ylabel = "Mix number",
              transparent = False,
              ylimit = (),
              xlimit = ()):
    plt.figure()
    ttl = plt.title("Mix number at various flow rate")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #xaxis.get_label()
    #yaxis.get_label()
    #get_xaxis().tick_bottom()    
    #get_yaxis().tick_left()    
    #plt.show()
    #plt.legend()
    #plt.savefig(outpath)
    if ylimit != ():
        plt.ylim(ylimit)
    if xlimit != ():
        plt.xlim(xlimit)
    for mix_number, time_data, color, label, pointtype, marker in zip(mixnumber, timedata, colors, labels, point_types, markers):
        ydata =  mix_number
        xdata = time_data
        ax= plt.subplot(111)
        plt.plot(xdata, ydata, color+marker, label = label, ms=3,markevery=20)
        
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)
    plt.legend()
    ax.grid('on')
    ttl = ax.title
    ttl.set_weight('bold')
    plt.savefig(outpath + 'plot' + '.svg', transparent = False, bbox_inches="tight")
    plt.show()
    #return plt.plot(xdata, ydata, color+ pointtype, label = label)
plotFigures = multiplot(MixNumber, timeScaled,
                        out_dir,
                        colors,
                        pointtypes,
                        labels,
                        markers,
                        xlabel = "Dimensionless time",
                        ylabel = "Mix number",
                        ylimit =(-0.5,1),
                        xlimit = (-0.3, 1.2) )
def scaledTem(listOfDfs_scaledTime):
    denominator = [p.max() - p.min() for p in listOfDfs_scaledTime]
    nominator = [p - p.min() for p in listOfDfs_scaledTime]
    scaled_value = [(n/d) for n,d  in zip (nominator,denominator)]
    return scaled_value
temScaled = scaledTem(string2float)
print (temScaled)
print (len(temScaled))
print (type(temScaled))

print (timeScaled)
print (len(timeScaled))
print (type(timeScaled))
print (len(temProfileColors))
print (type(temProfileColors))
for x in timeScaled:
    print (type(x))
##def igenerator(arg):
##    for i in range(0, len(arg)):
##        return i
##        return i
##        return i
##igen = igenerator(litersPerHour)
    
def create_color_step_obj(cmap_name, n):
    """
    Return scalarMap object with n colors in gradient from color map
    given in cmap_name.
    """
    cmap = plt.get_cmap(cmap_name)
    values = range(n)
    cNorm  = clrs.Normalize(vmin=values[0], vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    return(scalarMap)
    
def temProfile(listOfDfs_scaledTime,
               listOfDfs_tem,
               markers,
               labels,
               litersPerHours,
               cmap_name = "jet",
               alpha = 0.7,
               xlabel = "Dimensionless time",
              ylabel = "Dimensionless Temperature"):
    plt.figure()
    #ttl = plt.title("Temperature profile")
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    scm = create_color_step_obj(cmap_name, len(markers))
    for j, (dfs_time, dfs_tem, litersPerHour) in enumerate (zip(listOfDfs_scaledTime, listOfDfs_tem, litersPerHours)):
        ttl = plt.title("Temperature profile for " + litersPerHour)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        xdata = dfs_time
        ydata = dfs_tem
        for i in range(0, 20):
            plt.plot(xdata, ydata.iloc[:,i], color= scm.to_rgba(i, alpha), marker = markers[i],label = labels[i], ms = 5, markevery = 20)
        #j = [i for i in range(0, len(litersPerHour))]
    
        #plt.legend()
        plt.legend(loc=1, fontsize = 'xx-small')
        plt.savefig(out_dirs[j] + 'plot' + '.svg', transparent = False)
        plt.show()
TemProfile = temProfile(timeScaled,
                        temScaled,
                        listMarkers,
                        reversedTemlegends,
                        TemProfileLitersPerHour,
                        cmap_name = "jet",
                        alpha =0.7,
                        xlabel = "Dimensionless time",
                        ylabel = "Dimensionless Temperature")

##def createColorArrange_hot(n):
##    return [cm]

#for i, (dfs_time, dfs_tem, litersPerHour) in enumerate (zip(timeScaled, temScaled, TemProfileLitersPerHour)):
#color = colors[i]
        
##def temProfile(listOfDfs_scaledTime, listOfDfs_tem, colorss, markerss):
##    plt.figure()
##    for x, y in zip (listOfDfs_scaledTime, listOfDfs_tem):
##        #ydata =  y
##        #xdata = x
##        for color, marker in zip (colors, markers):
##            ydata =  y
##            xdata = x
##            color =color
##            marker = marker
##        plt.plot(xdata, ydata , color+marker, ms = 1, markevery=5)
##        plt.show()
##TemProfile = temProfile(timeScaled, temScaled,listColors,listMarkers)
##def temProfile(listOfDfs_scaledTime, listOfDfs_tem, colorss, markerss):
##    plt.figure()
##    for color, marker in zip (colors, markers):
##    
##        for x, y in zip (listOfDfs_scaledTime, listOfDfs_tem):
##            ydata =  y
##            xdata = x
##            #for color, marker in zip (colors, markers):
##            #ydata =  y
##            #xdata = x
##            color =color
##            marker = marker
##            plt.plot(xdata, ydata , color+marker, ms = 1, markevery=5)
##            plt.show()
##TemProfile = temProfile(timeScaled, temScaled,listColors,listMarkers)
##def temProfile(listOfDfs_scaledTime, listOfDfs_tem, colorss, markerss):
##    plt.figure()
##    for x, y in zip (listOfDfs_scaledTime, listOfDfs_tem):
##        for i,j, color,marker in zip (x,y, colorss, markerss):
##            ydata =  y
##            xdata = x
##            #for color, marker in zip (colors, markers):
##            #ydata =  y
##            #xdata = x
##            #color =color
##            #marker = marker
##            plt.plot(j,i , color+marker, ms = 1, markevery=5)
##            plt.show()
##TemProfile = temProfile(timeScaled, temScaled,listColors,listMarkers)

    
############################################
##End
############################################


