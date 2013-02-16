import matplotlib.pyplot as plt
import numpy as np

import os as os
import pylab as plb
import wave as wv
import matplotlib.mlab as mlab
import scipy.stats as stats
from numpy.core.defchararray import endswith

os.chdir('/Users/brianschmidt/Dropbox/Python/')

##################
### USER INPUT ###
##################

# BIRD INFO #
BIRDNAME = 'Bird 1815'
CONDITION = 0 ## 0 = Baseline, 1 = Post Manipulation
ANALYSIS = 1 ## Automatically compare Exp Condition to Baseline? 0 = No, 1 = Yes
DAY = 'Feb 2' ## Not important if Condition is 0
SYLLABLES = 5 ## Number of syllables to be analyzed

# DISPLAY OPTIONS #
DISPLAYSYLL = 0 ## Control whether syllables are displayed: 0=No, 1=Yes
DISPLAYSONG = 0 ## Control whether songs are displayed: 0=No, 1=Yes

# ANALYSIS OPTIONS #
MAXFILE = 2 ## Maximum number of songs to be analyzed
SAVESYLL = 0 ## 0 = Do Not save syllable data into csv files, 1 = Yes




########################
##### SET VARIABLES ####
########################
if CONDITION == 0:
    FILEDIR = (BIRDNAME+'/baseline song/')
elif CONDITION == 1:
    FILEDIR = (BIRDNAME+'/post-surgery song/'+DAY+'/')
files = np.array(os.listdir(FILEDIR))
wav_index = mlab.find(endswith(files,'.wav'))
files = files[wav_index]
DIST_NAME1 = ('DistABC_data.png')
DIST_NAME2 = ('DistDEF_data.png')
DUR_NAME = ('Duration.png')
MINFREQ = 1250
STARTFILE = 0
SONGPERCENT = 0.5 ## Fraction of song required for extraction


#############################################
#################SUBFUNCTIONS################
#############################################

def runs(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    # eliminate noise or short duration chirps
    length = run_ends-run_starts
    threshold = length>20
    return run_starts[threshold], run_ends[threshold]

def measure(sample, time, freqs):
    # Create matrix
    W = np.zeros((len(time),1))
    mean_freq = np.zeros((len(time),1))
    # calculate Weiner Entropy and Mean Freq
    for j in range(0,len(time)):
        w = np.log((np.exp(sum(np.log(sample[:,j]))/len(sample[:,j])))/(sum(sample[:,j])
                                                                        /len(sample[:,j])))
        fr = sum(freqs[winFreq]*np.power(sample[:,j],2))/sum(np.power(sample[:,j],2))
        W[j]= w
        mean_freq[j] = fr
    # Duration - in seconds
    duration = max(t[time])-min(t[time])
    return W, mean_freq, duration

def spec(file):
    spf = wv.open(file,'r')
    sound_info = spf.readframes(-1)
    sound_info = plb.fromstring(sound_info, 'Int16')
    f = spf.getframerate()
    plt.figure(num=None, figsize=(16, 7), dpi=80, edgecolor='k')
    p, freqs, t, im = plb.specgram(sound_info, Fs = f, scale_by_freq=True,sides='default',)
    plt.hold(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    font = {'size'  : 18}
    plt.rc('font', **font)
    plt.bbox_inches='tight'
    plt.tight_layout()
    plt.show()
    return p,freqs,t,im


def FindSong(p,q,t):
    l = p[winFreq,:]        
    l = l.mean(0)
    l = l>THRESHOLD
    q = q>THRESHOLD
    thresh = sum(q)
    if len(t)>len(q):
        score = np.zeros((len(t)-len(q),1))
        for i in range(0,len(t)-len(q)):
            sc = sum(l[i:i+len(q)])
            score[i] = sc

        difs = np.diff(score.flatten(0))
        peaks = np.r_[True,difs[:-1] > difs[1:]]
        peaks = np.where(peaks > 0)
        peak = np.where(score[peaks]>thresh*SONGPERCENT)
        song = peaks[0][peak[0]]
    elif len(t)<=len(q):
        song = np.array([])
    if not song.any():
        SongPower = False;SongTime=False
        return SongPower,SongTime
    elif song.any():
        ss = score[song].argmax()
        song = song[ss]
        SongTime = np.arange(song,song+(len(q))+20)
        if SongTime.max() > len(p[0,:]):
            SongPower = False;SongTime=False
            return SongPower,SongTime
        elif SongTime.max() <= len(p[0,:]):
            P = p[:,SongTime]
            SongPower = P[winFreq,:]
            if DISPLAYSONG == 1:
                Z = 10*np.log10(abs(SongPower))
                plt.figure()
                plt.imshow(Z,origin='lower', aspect='auto')
                plt.hold(True)
                plt.show()
            return SongPower,SongTime
        
############################################
################Create Files################
############################################
        
NAME = ['A','B','C','D','E','F']

os.makedirs(FILEDIR +'/Analysis')
if SAVESYLL == 1:
    os.makedirs(FILEDIR +'/SyllCSV')

# Set Syllable Files:
if SYLLABLES >= 1:
    SyllA_duration = np.zeros((len(files)*4,1))
    SyllA_entropy = np.zeros((len(files)*4,200))
    SyllA_freq = np.zeros((len(files)*4,200))
if SYLLABLES >= 2:
    SyllB_duration = np.zeros((len(files)*4,1))
    SyllB_entropy = np.zeros((len(files)*4,200))
    SyllB_freq = np.zeros((len(files)*4,200))
if SYLLABLES >= 3:
    SyllC_duration = np.zeros((len(files)*5,1))
    SyllC_entropy = np.zeros((len(files)*4,200))
    SyllC_freq = np.zeros((len(files)*4,200))
if SYLLABLES >= 4:
    SyllD_duration = np.zeros((len(files)*5,1))
    SyllD_entropy = np.zeros((len(files)*4,200))
    SyllD_freq = np.zeros((len(files)*4,200))
if SYLLABLES >= 5:
    SyllE_duration = np.zeros((len(files)*5,1))
    SyllE_entropy = np.zeros((len(files)*4,200))
    SyllE_freq = np.zeros((len(files)*4,200))
if SYLLABLES >= 6:
    SyllF_duration = np.zeros((len(files)*5,1))
    SyllF_entropy = np.zeros((len(files)*4,200))
    SyllF_freq = np.zeros((len(files)*4,200))

if SYLLABLES >= 1:
    SyllA = [SyllA_duration, SyllA_entropy, SyllA_freq]
if SYLLABLES >= 2:
    SyllB = [SyllB_duration, SyllB_entropy, SyllB_freq]
if SYLLABLES >= 3:
    SyllC = [SyllC_duration, SyllC_entropy, SyllC_freq]
if SYLLABLES >= 4:
    SyllD = [SyllD_duration, SyllD_entropy, SyllD_freq]
if SYLLABLES >= 5:
    SyllE = [SyllE_duration, SyllE_entropy, SyllE_freq]
if SYLLABLES >= 6:
    SyllF = [SyllF_duration, SyllF_entropy, SyllF_freq]

if SYLLABLES is 1:
    Syll = [SyllA]
if SYLLABLES is 2:
    Syll = [SyllA, SyllB]
if SYLLABLES is 3:
    Syll = [SyllA, SyllB, SyllC]
if SYLLABLES is 4:
    Syll = [SyllA, SyllB, SyllC, SyllD]
if SYLLABLES is 5:
    Syll = [SyllA, SyllB, SyllC, SyllD, SyllE]
if SYLLABLES is 6:
    Syll = [SyllA, SyllB, SyllC, SyllD, SyllE, SyllF]
    

###################
### Define Song ###
###################
    
def tellme(s):
    print s
    plt.title(s,fontsize=16)
    plt.draw()
            
for ii in range(STARTFILE,len(wav_index)):
    file = (FILEDIR+files[ii])    
    p,freqs,t,im = spec(file)
    p = p
    tellme('Select Song')
    plt.waitforbuttonpress()
    tellme('Select Song')
    happy = False
    Proceed = False
    while not happy:
        happy = plt.waitforbuttonpress()

        if not happy:
            tim = np.asarray( plt.ginput(n=0,timeout=-1) )
            ind = np.logical_and(t>tim[0,0], t<tim[1,0])
            SongTime = mlab.find(ind)
# Generate syllable spectrogram
            winFreq = freqs>MINFREQ
            winFreq = mlab.find(winFreq)
            P = p[:,SongTime]
            P = P[winFreq,:]
            plt.close(1)
            FIND_SONG = P.mean(0)
            THRESHOLD = min(FIND_SONG)*10
            Proceed = True
            break
            
        if happy:
            plt.close()
    if Proceed:
        break
    
###########################################
###########MAIN LOOP#######################
###########################################

if len(files) < MAXFILE+1:
    ENDFILE = len(files)
elif len(files) >= MAXFILE+1:
    ENDFILE = MAXFILE+1 
    
ii = STARTFILE
while ii < ENDFILE:
    file = (FILEDIR+files[ii])
    print ii
    
## Spectrogram ##

    p,freqs,t,im = spec(file)
    p = p*0.9
    SongP,SongTime = FindSong(p,FIND_SONG,t)
    if SongP is False:
        plt.close(1)
        if len(files) < MAXFILE+1:
            ENDFILE = len(files)
        elif len(files) >= MAXFILE+1 and ENDFILE < len(files):
            ENDFILE = ENDFILE+1
        elif ENDFILE >= len(files):
            ENDFILE = ENDFILE
        print 'Skipped because there was no song'
        pass
    ############# Find Syllables################
    elif SongP.any():
        mean_freq = np.zeros((len(SongTime),1))
        for j in range(0,len(SongTime)):        
            fr = sum(freqs[winFreq]*np.power(SongP[:,j],2))/sum(np.power(SongP[:,j],2))
            mean_freq[j] = fr
        del(fr); del(j);           
        q = SongP.mean(0)
        q = q>THRESHOLD
        tt = runs(q)
        FOUND = tt[0].any()
        if FOUND:
            count = 0
            for i in range(0,len(tt[0])):
                time = np.arange(tt[0][i],tt[1][i])
                if len(time) > 200:
                    plt.close(1)
                    plt.close(2)
                    print 'Skipped syllable because it was too long'
                    pass
                elif len(time) <= 200:
                    P = SongP[:,time]
                    W,mean_freq,duration = measure(P, time, freqs)
                    count = count+1
                    if count<=SYLLABLES:
                        # Generate syllable spectrogram
                        if DISPLAYSYLL == 1:
                            Z = 10*np.log10(abs(P))
                            plt.figure()
                            plt.imshow(Z,origin='lower', aspect='auto')
                            plt.hold(True)
                            plt.show()
                            plt.tight_layout()
                            plt.close(3)
                        if SAVESYLL == 1:
                            np.savetxt(FILEDIR+'SyllCSV/'+files[ii]+NAME[i]+'.csv', P,
                                       delimiter=",")

                    # Totals:
                        #Find correct file:
                        CurrSyll = Syll[count-1]
                        CurrSyllDur = CurrSyll[0]
                        CurrSyllEntropy = CurrSyll[1]
                        CurrSyllFreq = CurrSyll[2]

                        #Write data into files:
                        if CurrSyllDur[0] == 0:
                            x_place = 0
                        elif CurrSyllDur[0] != 0:
                            x_place = np.max(np.nonzero(CurrSyllDur[:,0]))+1
                        for i in range(0,len(time)):
                            CurrSyllFreq[x_place,i] = mean_freq[i]
                            CurrSyllEntropy[x_place,i] = W[i]
                        CurrSyllDur[x_place] = duration
                        plt.close(2)
                plt.close(1)
        if not FOUND:
            plt.close(1)
            plt.close(2)
            if len(files) < MAXFILE+1:
                ENDFILE = len(files)
            elif len(files) >= MAXFILE+1 and ENDFILE < len(files):
                ENDFILE = ENDFILE+1
            elif ENDFILE >= len(files):
                ENDFILE = ENDFILE
                print 'Skipped because there were no syllables extracted'           
    ii = ii+1

#####################
## Eliminate Zeros:##
#####################
if SYLLABLES >= 1:
    SyllA_duration = SyllA_duration[SyllA_duration.all(1)]
    SyllA_entropy = SyllA_entropy[:,np.any(SyllA_entropy,0)]
    SyllA_entropy = SyllA_entropy[np.any(SyllA_entropy,1),:]
    SyllA_entropy[SyllA_entropy==0] = np.NaN
    SyllA_freq = SyllA_freq[:,np.any(SyllA_freq,0)]
    SyllA_freq = SyllA_freq[np.any(SyllA_freq,1),:]
    SyllA_freq[SyllA_freq==0] = np.NaN
if SYLLABLES >= 2:
    SyllB_duration = SyllB_duration[SyllB_duration.all(1)]
    SyllB_entropy = SyllB_entropy[:,np.any(SyllB_entropy,0)]
    SyllB_entropy = SyllB_entropy[np.any(SyllB_entropy,1),:]
    SyllB_entropy[SyllB_entropy==0] = np.NaN
    SyllB_freq = SyllB_freq[:,np.any(SyllB_freq,0)]
    SyllB_freq = SyllB_freq[np.any(SyllB_freq,1),:]
    SyllB_freq[SyllB_freq==0] = np.NaN
if SYLLABLES >= 3:
    SyllC_duration = SyllC_duration[SyllC_duration.all(1)]
    SyllC_entropy = SyllC_entropy[:,np.any(SyllC_entropy,0)]
    SyllC_entropy = SyllC_entropy[np.any(SyllC_entropy,1),:]
    SyllC_entropy[SyllC_entropy==0] = np.NaN
    SyllC_freq = SyllC_freq[:,np.any(SyllC_freq,0)]
    SyllC_freq = SyllC_freq[np.any(SyllC_freq,1),:]
    SyllC_freq[SyllC_freq==0] = np.NaN
if SYLLABLES >= 4:
    SyllD_duration = SyllD_duration[SyllD_duration.all(1)]
    SyllD_entropy = SyllD_entropy[:,np.any(SyllD_entropy,0)]
    SyllD_entropy = SyllD_entropy[np.any(SyllD_entropy,1),:]
    SyllD_entropy[SyllD_entropy==0] = np.NaN
    SyllD_freq = SyllD_freq[:,np.any(SyllD_freq,0)]
    SyllD_freq = SyllD_freq[np.any(SyllD_freq,1),:]
    SyllD_freq[SyllD_freq==0] = np.NaN
if SYLLABLES >= 5:
    SyllE_duration = SyllE_duration[SyllE_duration.all(1)]
    SyllE_entropy = SyllE_entropy[:,np.any(SyllE_entropy,0)]
    SyllE_entropy = SyllE_entropy[np.any(SyllE_entropy,1),:]
    SyllE_entropy[SyllE_entropy==0] = np.NaN
    SyllE_freq = SyllE_freq[:,np.any(SyllE_freq,0)]
    SyllE_freq = SyllE_freq[np.any(SyllE_freq,1),:]
    SyllE_freq[SyllE_freq==0] = np.NaN
if SYLLABLES >= 6:
    SyllF_duration = SyllF_duration[SyllF_duration.all(1)]
    SyllF_entropy = SyllF_entropy[:,np.any(SyllF_entropy,0)]
    SyllF_entropy = SyllF_entropy[np.any(SyllF_entropy,1),:]
    SyllF_entropy[SyllF_entropy==0] = np.NaN
    SyllF_freq = SyllF_freq[:,np.any(SyllF_freq,0)]
    SyllF_freq = SyllF_freq[np.any(SyllF_freq,1),:]
    SyllF_freq[SyllF_freq==0] = np.NaN

#########################
#### Create .CSV Files:##
#########################

if SYLLABLES >= 1:
    # Creat file name Syllable A
    entropy = (FILEDIR+'Analysis/SyllA_entropyAnalysis.csv')
    duration = (FILEDIR+'Analysis/SyllA_durationAnalysis.csv')
    freq = (FILEDIR+'Analysis/SyllA_freqAnalysis.csv')

    # Write to excel file Syllable A:
    np.savetxt(entropy, SyllA_entropy, delimiter=",")
    np.savetxt(duration, SyllA_duration, delimiter=",")
    np.savetxt(freq, SyllA_freq, delimiter=",")
if SYLLABLES >= 2:
    # Creat file name Syllable B
    entropyB = (FILEDIR+'Analysis/SyllB_entropyAnalysis.csv')
    durationB = (FILEDIR+'Analysis/SyllB_durationAnalysis.csv')
    freqB = (FILEDIR+'Analysis/SyllB_freqAnalysis.csv')

    # Write to excel file Syllable B:
    np.savetxt(entropyB, SyllB_entropy, delimiter=",")
    np.savetxt(durationB, SyllB_duration, delimiter=",")
    np.savetxt(freqB, SyllB_freq, delimiter=",")
if SYLLABLES >= 3:	
    # Creat file name Syllable C
    entropyC = (FILEDIR+'Analysis/SyllC_entropyAnalysis.csv')
    durationC = (FILEDIR+'Analysis/SyllC_durationAnalysis.csv')
    freqC = (FILEDIR+'Analysis/SyllC_freqAnalysis.csv')

    # Write to excel file Syllable C:
    np.savetxt(entropyC, SyllC_entropy, delimiter=",")
    np.savetxt(durationC, SyllC_duration, delimiter=",")
    np.savetxt(freqC, SyllC_freq, delimiter=",")
if SYLLABLES >= 4:
    # Creat file name Syllable D
    entropyD = (FILEDIR+'Analysis/SyllD_entropyAnalysis.csv')
    durationD = (FILEDIR+'Analysis/SyllD_durationAnalysis.csv')
    freqD = (FILEDIR+'Analysis/SyllD_freqAnalysis.csv')

    # Write to excel file Syllable D:
    np.savetxt(entropyD, SyllD_entropy, delimiter=",")
    np.savetxt(durationD, SyllD_duration, delimiter=",")
    np.savetxt(freqD, SyllD_freq, delimiter=",")
if SYLLABLES >= 5:
    # Creat file name Syllable E
    entropyE = (FILEDIR+'Analysis/SyllE_entropyAnalysis.csv')
    durationE = (FILEDIR+'Analysis/SyllE_durationAnalysis.csv')
    freqE = (FILEDIR+'Analysis/SyllE_freqAnalysis.csv')

    # Write to excel file Syllable E:
    np.savetxt(entropyE, SyllE_entropy, delimiter=",")
    np.savetxt(durationE, SyllE_duration, delimiter=",")
    np.savetxt(freqE, SyllE_freq, delimiter=",")
if SYLLABLES >= 6:
    # Creat file name Syllable F
    entropyF = (FILEDIR+'Analysis/SyllF_entropyAnalysis.csv')
    durationF = (FILEDIR+'Analysis/SyllF_durationAnalysis.csv')
    freqF = (FILEDIR+'Analysis/SyllF_freqAnalysis.csv')

    # Write to excel file Syllable F:
    np.savetxt(entropyF, SyllF_entropy, delimiter=",")
    np.savetxt(durationF, SyllF_duration, delimiter=",")
    np.savetxt(freqF, SyllF_freq, delimiter=",")

#################
## Distribution##
#################


FSize=15
plt.figure(figsize=(11,9))
font = {'weight' : 'norm','size'  : 16}
plt.rc('font', **font)
FREQBIN = 100
DE_X = np.arange(-14,0.1,0.25)
DF_X = np.arange(MINFREQ,np.max(freqs),FREQBIN)
labelx = -0.21  # axes coords

#Syllable A:
if SYLLABLES >= 1:
    #Entropy
    DistA_entropy = SyllA_entropy.flatten(1)
    DistA_entropy[~np.isnan(DistA_entropy)]
    a = plt.subplot(321)
    DAE,DAE_X = np.histogram(DistA_entropy, bins=DE_X, density=False)
    a.bar(np.arange(-14,0,0.25),np.true_divide(DAE,np.sum(DAE)), width=0.2, color='r')
    plb.ylabel('Proportion', fontsize= FSize)
    a.set_xlim(-14,0)
    a.yaxis.set_label_coords(labelx, 0.5)
    plb.title('Syllable A', fontsize= FSize)

    #Frequency    
    DistA_freq = SyllA_freq.flatten(1)
    DistA_freq = DistA_freq[~np.isnan(DistA_freq)]
    b = plt.subplot(322)
    DAF,DAF_X = np.histogram(DistA_freq, bins=DF_X, density=False)
    b.bar(np.arange(MINFREQ,np.max(freqs)-FREQBIN,FREQBIN),np.true_divide(DAF,np.sum(DAF)),
          width=50, color='r',edgecolor=None)
    b.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), ['0','1','2','3','4','5','6','7','8'])
    plb.title('Syllable A', fontsize= FSize)
    
#Syllable B:
if SYLLABLES >= 2:
    #Entropy
    DistB_entropy = SyllB_entropy.flatten(1)
    DistB_entropy = DistB_entropy[~np.isnan(DistB_entropy)]
    c = plt.subplot(323)
    DBE,DBE_X = np.histogram(DistB_entropy, bins=DE_X , density=False)
    c.bar(np.arange(-14,0,0.25),np.true_divide(DBE,np.sum(DBE)), width=0.2, color='b')
    plb.ylabel('Proportion', fontsize= FSize)
    c.yaxis.set_label_coords(labelx, 0.5)
    c.set_xlim(-14,0)
    plb.title('Syllable B', fontsize= FSize)

    #Frequency    
    DistB_freq = SyllB_freq.flatten(1)
    DistB_freq = DistB_freq[~np.isnan(DistB_freq)]
    d = plt.subplot(324)
    DBF,DBF_X = np.histogram(DistB_freq, bins=DF_X, density=False)
    d.bar(np.arange(MINFREQ,np.max(freqs)-FREQBIN,FREQBIN),np.true_divide(DBF,np.sum(DBF)),
          width=50, color='b',edgecolor=None)
    d.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), ['0','1','2','3','4','5','6','7','8'])
    plb.title('Syllable B', fontsize= FSize)
    
#Syllable C:
if SYLLABLES >= 3:
    #Entropy
    DistC_entropy = SyllC_entropy.flatten(1)
    DistC_entropy = DistC_entropy[~np.isnan(DistC_entropy)]
    e = plt.subplot(325)
    DCE,DCE_X = np.histogram(DistC_entropy, bins=DE_X, density=False)
    e.bar(np.arange(-14,0,0.25),np.true_divide(DCE,np.sum(DCE)), width=0.2, color='g')
    plb.ylabel('Proportion', fontsize= FSize)
    e.yaxis.set_label_coords(labelx, 0.5)
    plb.xlabel('Weiner Entropy', fontsize = FSize)
    e.set_xlim(-14,0)
    plb.title('Syllable C', fontsize= FSize)

    #Frequency    
    DistC_freq = SyllC_freq.flatten(1)
    DistC_freq = DistC_freq[~np.isnan(DistC_freq)]
    f = plt.subplot(326)
    DCF,DCF_X = np.histogram(DistC_freq, bins=DF_X, density=False)
    f.bar(np.arange(MINFREQ,np.max(freqs)-FREQBIN,FREQBIN),np.true_divide(DCF,np.sum(DCF)),
          width=50, color='g',edgecolor=None)
    plb.xlabel('Frequency (kHz)', fontsize = FSize)
    f.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), ['0','1','2','3','4','5','6','7','8'])
    plb.title('Syllable C', fontsize= FSize)

plt.subplots_adjust(wspace=0.21,hspace=0.308)
plt.show()
plb.savefig(FILEDIR+'Analysis/'+DIST_NAME1)

#### Plots D,E,F:
if SYLLABLES >= 4:
    plt.figure(figsize=(11,9))
    font = {'weight' : 'norm','size'  : 16}
    plt.rc('font', **font)

    #Syllable D:

    #Entropy
    DistD_entropy = SyllD_entropy.flatten(1)
    DistD_entropy[~np.isnan(DistD_entropy)]
    g = plt.subplot(321)
    DDE,DDE_X = np.histogram(DistD_entropy, bins=DE_X, density=False)
    g.bar(np.arange(-14,0,0.25),np.true_divide(DDE,np.sum(DDE)), width=0.2, color='r')
    plb.ylabel('Proportion', fontsize= FSize)
    g.set_xlim(-14,0)
    g.yaxis.set_label_coords(labelx, 0.5)
    plb.title('Syllable D', fontsize= FSize)

    if SYLLABLES is 4:
        plb.xlabel('Weiner Entropy', fontsize = FSize)

    #Frequency    
    DistD_freq = SyllD_freq.flatten(1)
    DistD_freq = DistD_freq[~np.isnan(DistD_freq)]
    h = plt.subplot(322)
    DDF,DDF_X = np.histogram(DistD_freq, bins=DF_X, density=False)
    h.bar(np.arange(MINFREQ,np.max(freqs)-FREQBIN,FREQBIN),np.true_divide(DDF,np.sum(DDF)),
          width=50, color='r',edgecolor=None)
    h.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), ['0','1','2','3','4','5','6','7','8'])
    plb.title('Syllable D', fontsize= FSize)

    if SYLLABLES is 4:
        plb.xlabel('Frequency (kHz)', fontsize = FSize)
        plt.subplots_adjust(wspace=0.21,hspace=0.308)
        plt.show()
        plb.savefig(FILEDIR+'Analysis/'+DIST_NAME2)
    
#Syllable E:
if SYLLABLES >= 5:
    #Entropy
    DistE_entropy = SyllE_entropy.flatten(1)
    DistE_entropy = DistE_entropy[~np.isnan(DistE_entropy)]
    i = plt.subplot(323)
    DEE,DEE_X = np.histogram(DistE_entropy, bins=DE_X , density=False)
    i.bar(np.arange(-14,0,0.25),np.true_divide(DEE,np.sum(DEE)), width=0.2, color='b')
    plb.ylabel('Proportion', fontsize= FSize)
    i.yaxis.set_label_coords(labelx, 0.5)
    i.set_xlim(-14,0)
    plb.title('Syllable E', fontsize= FSize)

    if SYLLABLES is 5:
        plb.xlabel('Weiner Entropy', fontsize = FSize)
        
    #Frequency    
    DistE_freq = SyllE_freq.flatten(1)
    DistE_freq = DistE_freq[~np.isnan(DistE_freq)]
    j = plt.subplot(324)
    DEF,DEF_X = np.histogram(DistE_freq, bins=DF_X, density=False)
    j.bar(np.arange(MINFREQ,np.max(freqs)-FREQBIN,FREQBIN),np.true_divide(DEF,np.sum(DEF)),
          width=50, color='b',edgecolor=None)
    j.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), ['0','1','2','3','4','5','6','7','8'])
    plb.title('Syllable E', fontsize= FSize)

    if SYLLABLES is 5:
        plb.xlabel('Frequency (kHz)', fontsize = FSize)
        plt.subplots_adjust(wspace=0.21,hspace=0.308)
        plt.show()
        plb.savefig(FILEDIR+'Analysis/'+DIST_NAME2)
#Syllable F:
if SYLLABLES >= 6:
    #Entropy
    DistF_entropy = SyllF_entropy.flatten(1)
    DistF_entropy = DistF_entropy[~np.isnan(DistF_entropy)]
    k = plt.subplot(325)
    DFE,DFE_X = np.histogram(DistF_entropy, bins=DE_X, density=False)
    k.bar(np.arange(-14,0,0.25),np.true_divide(DFE,np.sum(DFE)), width=0.2, color='g')
    plb.ylabel('Proportion', fontsize= FSize)
    k.yaxis.set_label_coords(labelx, 0.5)
    plb.xlabel('Weiner Entropy', fontsize = FSize)
    k.set_xlim(-14,0)
    plb.title('Syllable F', fontsize= FSize)

    #Frequency    
    DistF_freq = SyllF_freq.flatten(1)
    DistF_freq = DistF_freq[~np.isnan(DistF_freq)]
    l = plt.subplot(326)
    DFF,DFF_X = np.histogram(DistF_freq, bins=DF_X, density=False)
    l.bar(np.arange(MINFREQ,np.max(freqs)-FREQBIN,FREQBIN),np.true_divide(DFF,np.sum(DFF)),
          width=50, color='g',edgecolor=None)
    plb.xlabel('Frequency (kHz)', fontsize = FSize)
    l.set_xlim(np.min(freqs),8000)
    plt.xticks(np.arange(0,8001,1000), ['0','1','2','3','4','5','6','7','8'])
    plb.title('Syllable F', fontsize= FSize)

    plt.subplots_adjust(wspace=0.21,hspace=0.308)
    plt.show()
    plb.savefig(FILEDIR+'Analysis/'+DIST_NAME2)





##############
## Duration ##
##############
if SYLLABLES >= 1:
    Mean_SyllA = np.mean(SyllA_duration)
    StDv_SyllA = np.std(SyllA_duration)
if SYLLABLES >= 2:
    Mean_SyllB = np.mean(SyllB_duration)
    StDv_SyllB = np.std(SyllB_duration)
if SYLLABLES >= 3:
    Mean_SyllC = np.mean(SyllC_duration)
    StDv_SyllC = np.std(SyllC_duration)
if SYLLABLES >= 4:
    Mean_SyllD = np.mean(SyllD_duration)
    StDv_SyllD = np.std(SyllD_duration)
if SYLLABLES >= 5:
    Mean_SyllE = np.mean(SyllE_duration)
    StDv_SyllE = np.std(SyllE_duration)
if SYLLABLES >= 6:
    Mean_SyllF = np.mean(SyllF_duration)
    StDv_SyllF = np.std(SyllF_duration)


if SYLLABLES is 1:
    Y = [Mean_SyllA]
    X = [1]
    YERR = [StDv_SyllA]
    Lables = ['A']
if SYLLABLES is 2:
    Y = [Mean_SyllA,Mean_SyllB]
    X = [1,2]
    YERR = [StDv_SyllA, StDv_SyllB]
    Lables = ['A','B']
if SYLLABLES is 3:
    Y = [Mean_SyllA,Mean_SyllB,Mean_SyllC]
    X = [1,2,3]
    YERR = [StDv_SyllA, StDv_SyllB, StDv_SyllC]
    Lables = ['A','B', 'C']
if SYLLABLES is 4:
    Y = [Mean_SyllA,Mean_SyllB,Mean_SyllC,Mean_SyllD]
    X = [1,2,3,4]
    YERR = [StDv_SyllA, StDv_SyllB, StDv_SyllC, StDv_SyllD]
    Lables = ['A','B', 'C', 'D']
if SYLLABLES is 5:
    Y = [Mean_SyllA,Mean_SyllB,Mean_SyllC,Mean_SyllD,Mean_SyllE]
    X = [1,2,3,4,5]
    YERR = [StDv_SyllA, StDv_SyllB, StDv_SyllC, StDv_SyllD, StDv_SyllE]
    Lables = ['A','B','C','D','E']
if SYLLABLES is 6:
    Y = [Mean_SyllA,Mean_SyllB,Mean_SyllC,Mean_SyllD,Mean_SyllE,Mean_SyllF]
    X = [1,2,3,4,5,6]
    YERR = [StDv_SyllA, StDv_SyllB, StDv_SyllC, StDv_SyllD, StDv_SyllE, StDv_SyllF]
    Lables = ['A','B','C', 'D', 'E', 'F']

plt.figure()
plt.bar(X,Y, color='w', linewidth=3, align='center',
        yerr=YERR, ecolor='k',capsize=10)
plb.title('Duration', fontsize= 20)
plb.ylabel('Syllable Length (s)', fontsize= 20)
plt.yticks(size='large')
plt.xticks(np.arange(1,SYLLABLES+1),Lables, size='large')
plt.xlabel('Syllable', fontsize=20)
plt.tight_layout()

plt.show()
plb.savefig(FILEDIR+'Analysis/'+DUR_NAME)



###############BASELINE###############

##READ FILES##
if CONDITION == 1 & ANALYSIS == 1:
    FILEDIR_BASE = (BIRDNAME+'/baseline song/Analysis/')
    
    FILE = ['SyllA_durationAnalysis.csv','SyllA_entropyAnalysis.csv','SyllA_freqAnalysis.csv',
            'SyllB_durationAnalysis.csv','SyllB_entropyAnalysis.csv','SyllB_freqAnalysis.csv',
            'SyllC_durationAnalysis.csv','SyllC_entropyAnalysis.csv','SyllC_freqAnalysis.csv',
            'SyllD_durationAnalysis.csv','SyllD_entropyAnalysis.csv','SyllD_freqAnalysis.csv',
            'SyllE_durationAnalysis.csv','SyllE_entropyAnalysis.csv','SyllE_freqAnalysis.csv',
            'SyllF_durationAnalysis.csv','SyllF_entropyAnalysis.csv','SyllF_freqAnalysis.csv']

    DATA = ['SyllA_duration', 'SyllA_entropy', 'SyllA_freq',
            'SyllB_duration', 'SyllB_entropy', 'SyllB_freq',
            'SyllC_duration', 'SyllC_entropy', 'SyllC_freq',
            'SyllD_duration', 'SyllD_entropy', 'SyllD_freq',
            'SyllE_duration', 'SyllE_entropy', 'SyllE_freq',
            'SyllF_duration', 'SyllF_entropy', 'SyllF_freq']
           
    for i in range (0,SYLLABLES*3):
        DATA[i] = np.genfromtxt(FILEDIR_BASE+FILE[i], delimiter=',')

    if SYLLABLES >= 1:
        BaseSyllA_duration = DATA[0]
        BaseSyllA_entropy = DATA[1]
        BaseSyllA_freq = DATA[2]
        #Entropy
        BaseDistA_entropy = BaseSyllA_entropy.flatten(1)
        BaseDistA_entropy = BaseDistA_entropy[~np.isnan(BaseDistA_entropy)]
        #Frequency    
        BaseDistA_freq = BaseSyllA_freq.flatten(1)
        BaseDistA_freq = BaseDistA_freq[~np.isnan(BaseDistA_freq)]

    if SYLLABLES >= 2:
        BaseSyllB_duration = DATA[3]
        BaseSyllB_entropy = DATA[4]
        BaseSyllB_freq = DATA[5]
        #Entropy
        BaseDistB_entropy = BaseSyllB_entropy.flatten(1)
        BaseDistB_entropy = BaseDistB_entropy[~np.isnan(BaseDistB_entropy)]
        #Frequency    
        BaseDistB_freq = BaseSyllB_freq.flatten(1)
        BaseDistB_freq = BaseDistB_freq[~np.isnan(BaseDistB_freq)]
        
    if SYLLABLES >= 3:
        BaseSyllC_duration = DATA[6]
        BaseSyllC_entropy = DATA[7]
        BaseSyllC_freq = DATA[8]
        #Entropy
        BaseDistC_entropy = BaseSyllC_entropy.flatten(1)
        BaseDistC_entropy = BaseDistC_entropy[~np.isnan(BaseDistC_entropy)]
        #Frequency    
        BaseDistC_freq = BaseSyllC_freq.flatten(1)
        BaseDistC_freq = BaseDistC_freq[~np.isnan(BaseDistC_freq)]
                                      
    if SYLLABLES >= 4:
        BaseSyllD_duration = DATA[9]
        BaseSyllD_entropy = DATA[10]
        BaseSyllD_freq = DATA[11]
        #Entropy
        BaseDistD_entropy = BaseSyllD_entropy.flatten(1)
        BaseDistD_entropy = BaseDistD_entropy[~np.isnan(BaseDistD_entropy)]
        #Frequency    
        BaseDistD_freq = BaseSyllD_freq.flatten(1)
        BaseDistD_freq = BaseDistD_freq[~np.isnan(BaseDistD_freq)]
                                     
    if SYLLABLES >= 5:
        BaseSyllE_duration = DATA[12]
        BaseSyllE_entropy = DATA[13]
        BaseSyllE_freq = DATA[14]
        #Entropy
        BaseDistE_entropy = BaseSyllE_entropy.flatten(1)
        BaseDistE_entropy = BaseDistE_entropy[~np.isnan(BaseDistE_entropy)]
        #Frequency    
        BaseDistE_freq = BaseSyllE_freq.flatten(1)
        BaseDistE_freq = BaseDistE_freq[~np.isnan(BaseDistE_freq)]
                                      
    if SYLLABLES >= 6:
        BaseSyllF_duration = DATA[15]
        BaseSyllF_entropy = DATA[16]
        BaseSyllF_freq = DATA[17]
        #Entropy
        BaseDistF_entropy = BaseSyllF_entropy.flatten(1)
        BaseDistF_entropy = BaseDistF_entropy[~np.isnan(BaseDistF_entropy)]
        #Frequency    
        BaseDistF_freq = BaseSyllF_freq.flatten(1)
        BaseDistF_freq = BaseDistF_freq[~np.isnan(BaseDistF_freq)]

        ################
        ## KS/T STATS ##
        ################
    if SYLLABLES >= 1:
        AED,AEp = stats.ks_2samp(DistA_freq,BaseDistA_freq)
        AFD,AFp = stats.ks_2samp(DistA_entropy,BaseDistA_entropy)
        ADT,ADp = stats.ks_2samp(SyllA_duration.flatten(),BaseSyllA_duration)
    if SYLLABLES >= 2:
        BED,BEp = stats.ks_2samp(DistB_freq,BaseDistB_freq)
        BFD,BFp = stats.ks_2samp(DistB_entropy,BaseDistB_entropy)
        BDT,BDp = stats.ks_2samp(SyllB_duration.flatten(),BaseSyllB_duration)
    if SYLLABLES >= 3:
        CED,CEp = stats.ks_2samp(DistC_freq,BaseDistC_freq)
        CFD,CFp = stats.ks_2samp(DistC_entropy,BaseDistC_entropy)
        CDT,CDp = stats.ks_2samp(SyllC_duration.flatten(),BaseSyllC_duration)
    if SYLLABLES >= 4:
        DED,DEp = stats.ks_2samp(DistD_freq,BaseDistD_freq)
        DFD,DFp = stats.ks_2samp(DistD_entropy,BaseDistD_entropy)
        DDT,DDp = stats.ks_2samp(SyllD_duration.flatten(),BaseSyllD_duration)
    if SYLLABLES >= 5:
        EED,EEp = stats.ks_2samp(DistE_freq,BaseDistE_freq)
        EFD,EFp = stats.ks_2samp(DistE_entropy,BaseDistE_entropy)
        EDT,EDp = stats.ks_2samp(SyllE_duration.flatten(),BaseSyllE_duration)
    if SYLLABLES >= 6:
        FED,FEp = stats.ks_2samp(DistF_freq,BaseDistF_freq)
        FFD,FFp = stats.ks_2samp(DistF_entropy,BaseDistF_entropy)
        FDT,FDp = stats.ks_2samp(SyllF_duration.flatten(),BaseSyllF_duration)

    if SYLLABLES is 1:
        EntSTATS = [(AED)]
        FreqSTATS = [(AFD)]
        DurSTATS  = [(ADT)]
    if SYLLABLES is 2:
        EntSTATS = [(AED),(BED)]
        FreqSTATS = [(AFD),(BFD)]
        DurSTATS  = [(ADT),(BDT)]
    if SYLLABLES is 3:
        EntSTATS = [(AED),(BED),(CED)]
        FreqSTATS = [(AFD),(BFD),(CFD)]
        DurSTATS  = [(ADT),(BDT),(CDT)]
    if SYLLABLES is 4:
        EntSTATS = [(AED),(BED),(CED),(DED)]
        FreqSTATS = [(AFD),(BFD),(CFD),(DFD)]
        DurSTATS  = [(ADT),(BDT),(CDT),(DDT)]
    if SYLLABLES is 5:
        EntSTATS = [(AED),(BED),(CED),(DED),(EED)]
        FreqSTATS = [(AFD),(BFD),(CFD),(DFD),(EFD)]
        DurSTATS  = [(ADT),(BDT),(CDT),(DDT),(EDT)]
    if SYLLABLES is 6:
        EntSTATS = [(AED),(BED),(CED),(DED),(EED),(FED)]
        FreqSTATS = [(AFD),(BFD),(CFD),(DFD),(EFD),(FED)]
        DurSTATS  = [(ADT),(BDT),(CDT),(DDT),(EDT),(FDT)]

    # Write to excel file:
    np.savetxt(FILEDIR+'Analysis/'+'EntropyKS_STATS.csv', EntSTATS, delimiter=",")
    np.savetxt(FILEDIR+'Analysis/'+'FreqKS_STATS.csv', FreqSTATS, delimiter=",")
    np.savetxt(FILEDIR+'Analysis/'+'DurationKS_STATS.csv', DurSTATS, delimiter=",")
        

