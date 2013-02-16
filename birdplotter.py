# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np
import os as os

os.chdir('/Users/brianschmidt/Dropbox/Python/FinalSyllAnalysis/')

SYLLABLES = 5
BIRDNAME = 'Bird 1958'
DPS = ['2','4','6','8','10','12','14','16','18','20','22','24','26','23']
entropy = np.genfromtxt(BIRDNAME + '/entropy.csv', delimiter=',')
freq = np.genfromtxt(BIRDNAME + '/freq.csv', delimiter=',')
dur = np.genfromtxt(BIRDNAME + '/dur.csv', delimiter=',')
durA = np.multiply(dur**2,1000)
color = np.arange(0,SYLLABLES)
MEANCOLOR = 'b'

DIST_NAME1 = ('DistABC_data.png')
DIST_NAME2 = ('DistDEF_data.png')
DUR_NAME = ('Duration.png')


#################
## Distribution##
#################


FSize=15
plt.figure(figsize=(11,9))
font = {'weight': 'norm', 'size': 16}
plt.rc('font', **font)

labelx = -0.21  # axes coords

for syll in self.syllables:
    #Entropy
    ax1 = plt.subplot(211)
    pf.AxisFormat()
    pf.TufteAxis(ax1, ['left','bottom'])

    ax1.bar(np.arange(-14,0,0.25),np.true_divide(DAE,np.sum(DAE)), 
           width=0.2, color='r')
    ax1.set_ylabel('Proportion', fontsize= FSize)
    ax1.set_xlim(-14,0)
    ax1.yaxis.set_label_coords(labelx, 0.5)
    plt.title('Syllable ' + syll, fontsize= 20)

 
    ax2 = plt.subplot(212)   
    ax2.bar(np.arange(self.MINFREQ, np.max(freqs) - FREQBIN, FREQBIN), 
          np.true_divide(DAF,np.sum(DAF)), width=50, color='r',
                        edgecolor=None)
    ax2.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), 
               ['0','1','2','3','4','5','6','7','8'])
    plt.title('Syllable A', fontsize=FSize)

##############
## Duration ##
##############


plt.figure()
plt.bar(X,Y, color='w', linewidth=3, align='center',
        yerr=STDev, ecolor='k',capsize=10)
plt.title('Duration', fontsize= 20)
plt.ylabel('Syllable Length (s)', fontsize= 20)
plt.yticks(size='large')
#plt.xticks(np.arange(1,SYLLABLES+1), Lables, size='large')
plt.xlabel('Syllable', fontsize=20)

plt.tight_layout()
plt.show()
#plt.savefig(FILEDIR + 'Analysis/' + DUR_NAME)




########### Entropy #############

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(0, len(dur[:,0])):

    ax.plot(entropy[i,:], linewidth=3, marker='o', markersize=10)

ax.hold(True)
ax.set_xlabel('Day Post Surgery', fontsize=20)
ax.set_ylabel('KS Statistic', fontsize=20)
ax.set_title('Weiner Entropy', fontsize=20)
plt.ylim(0,1)
ax.grid(True)
plt.xticks(np.arange(0,len(dur[0,:])),DPS,size=20)
plt.yticks(size=20)
##plt.legend(('A','B','C','D','E'), loc='upper left')
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_Entropy.png')

########### Frequency #############

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(0, len(dur[:,0])):

    ax.plot(freq[i,:], linewidth=3, marker='o', markersize=10)

ax.hold(True)
ax.set_xlabel('Day Post Surgery', fontsize=20)
ax.set_ylabel('KS Statistic', fontsize=20)
ax.set_title('Frequency', fontsize=20)
plt.ylim(0,1)
ax.grid(True)
plt.xticks(np.arange(0,len(dur[0,:])),DPS,size=20)
plt.yticks(size=20)
##plt.legend(('A','B','C','D','E'), loc='upper left')
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_Freq.png')

########### Duration #############

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(0, len(dur[:,0])):

    ax.plot(dur[i,:], linewidth=3, marker='o', markersize=10)

ax.hold(True)
ax.set_xlabel('Day Post Surgery', fontsize=20)
ax.set_ylabel('KS Statistic', fontsize=20)
ax.set_title('Duration', fontsize=20)
##plt.ylim(0,0.8)
ax.grid(True)
plt.xticks(np.arange(0,len(dur[0,:])),DPS,size=20)
plt.yticks(size=20)
##plt.legend(('A','B','C','D','E'), loc='upper left')
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_Dur.png')


########### Scatter #############

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(0, len(dur[0,:])):

    ax.scatter(entropy[:,i], freq[:,i], c=['r','g','b','c','m'], s=durA[:,i], alpha=0.75)

ax.hold(True)
ax.set_xlabel('Entropy (KS Statistic)', fontsize=20)
ax.set_ylabel('Frequency (KS Statistic)', fontsize=20)
ax.set_title(BIRDNAME, fontsize=20)
plt.xlim(0,1)
plt.ylim(0,1)
ax.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_Scatter.png')

########### Euclidean Length #############

plt.figure()
Magnitude = np.zeros((SYLLABLES,len(dur[0,:])))

for j in range(0,SYLLABLES):
    for i in range(0,len(dur[0,:])):

        d = np.sqrt( ((dur[j,i]-0)**2) + ((entropy[j,i]-0)**2) + ((freq[j,i]-0)**2))

        Magnitude[j,i] = d
    plt.plot(Magnitude[j,:], linewidth=3, marker='o', markersize=10)

plt.xlabel('Day Post Surgery', fontsize=20)
plt.ylabel('Magnitude', fontsize=20)
plt.title('Song variance from Baseline', fontsize=20)
plt.grid(True)
plt.xticks(np.arange(0,len(dur[0,:])),DPS,size=20)
plt.yticks(size=20)
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_Mag.png')
np.savetxt(BIRDNAME+'/'+BIRDNAME+'Magnitude.csv', Magnitude, delimiter=",")

############ Mean Euclidean ################

meanMag = np.zeros((1,len(Magnitude[0,:])))
stdvMag = np.zeros((1,len(Magnitude[0,:])))
for i in range(0,len(Magnitude[0,:])):

    mean = np.mean(Magnitude[:,i])
    stdv = np.std(Magnitude[:,i])
    
    meanMag[:,i] = mean
    stdvMag[:,i] = stdv

plt.figure()
plt.plot(meanMag[0,:], MEANCOLOR, linewidth=4)
#plt.plot(meanMag+stdvMag, 'b')
#plt.plot(meanMag-stdvMag, 'b')
plt.xlabel('Day Post Surgery', fontsize=20)
plt.ylabel('Magnitude', fontsize=20)
plt.title('Song variance from Baseline', fontsize=20)
plt.grid(True)
plt.xlim(0,len(meanMag))
plt.fill_between(np.arange(0,len(meanMag[0,:])),meanMag[0,:], meanMag[0,:]+stdvMag[0,:],
                 facecolor=MEANCOLOR,alpha=0.5)
plt.fill_between(np.arange(0,len(meanMag[0,:])),meanMag[0,:], meanMag[0,:]-stdvMag[0,:],
                 facecolor=MEANCOLOR,alpha=0.5)
plt.xticks(np.arange(0,len(dur[0,:])),DPS,size=20)
plt.yticks(size=20)
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_MeanMag.png')


##############  Euclidean Delta ####################


plt.figure()
DeltaEuc = np.zeros((SYLLABLES,len(dur[0,:])-1))

for j in range(0,SYLLABLES):
    for i in range(0,len(dur[0,:])-1):

        d = np.sqrt( ((dur[j,i]-dur[j,i+1])**2) + ((entropy[j,i]-entropy[j,i+1])**2)
                     + ((freq[j,i]-freq[j,i+1])**2))

        DeltaEuc[j,i] = d
    plt.plot(DeltaEuc[j,:], linewidth=3, marker='o', markersize=10)

#plt.xlabel('Measurement Days', fontsize=20)
plt.ylabel('Magnitude', fontsize=20)
plt.title('Movement in Euclidean Space', fontsize=20)
plt.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()

plt.show()
plt.savefig(BIRDNAME+'/'+BIRDNAME + '_EucDelta.png')

