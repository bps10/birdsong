# -*- coding: utf-8 -*-
from __future__ import division
import os as os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from numpy.core.defchararray import endswith
from scipy import stats

import subfunctions as sf
import PlottingFun as pf


class SongAnalysis(object):
    """
    """
    
    def __init__(self, SYLLABLES, FILEDIR, BIRDNAME, CONDITION, DATE,
                 MAXFILE, MINFREQ=1250, STARTFILE=0,
                 DISPLAYSYLL=False, DISPLAYSONG=False):
        """
        """
        self.MINFREQ = MINFREQ
        self.MAXFILE = MAXFILE
        self.STARTFILE = STARTFILE
        self.BIRDNAME = BIRDNAME
        self.DATE = DATE
        self.FILEDIR = FILEDIR + '/'
        if CONDITION == 0:
            self.CONDITION = 'baseline'
        else:
            self.CONDITION = 'experiment'
            
        self.main(SYLLABLES, DISPLAYSYLL, DISPLAYSONG)

    def main(self, SYLLABLES, DISPLAYSYLL, DISPLAYSONG):
        """
        """
        
        self.getFiles()
        self.genDataStruct(SYLLABLES)
        self.findGoodSong()
        self.cutSyllables(SYLLABLES, DISPLAYSYLL, DISPLAYSONG)
        self.analysis()
        
        if self.CONDITION == 'experiment':
            self.findKSstat()
            self.distPlotter()
        self.savePickle()
        
    def getFiles(self):
        """
        """

        FILES = np.array(os.listdir(self.FILEDIR))
        self.wav_index = mlab.find(endswith(FILES,'.wav'))
        self.FILES = FILES[self.wav_index] 
        
    def genDataStruct(self, SYLLABLES):
        """
        """

        self.syllables = {}
        for i in range(SYLLABLES):
            self.syllables[i] =  {
                    'duration': {},
                    'entropy': {},
                    'freq': {},
                        }

    def saveMetaData(self):
        """
        """
        if self.CONDITION == 'experiment':
            birdmeta = self.loadPickle(name=self.BIRDNAME)
        else: 
            birdmeta = {}
        
        birdmeta[self.DATE] = {
                'MINFREQ': self.MINFREQ,
                'MAXFILE': self.MAXFILE,
                'STARTFILE': self.STARTFILE,
                'BIRDNAME': self.BIRDNAME,
                'DATE': self.DATE,
                'FILEDIR': self.FILEDIR,
        }
        
        self.savePickle(birdmeta)
        
        
    def findGoodSong(self):
        """
        """
        
        for i in range(self.STARTFILE, len(self.wav_index)):

            FILE = (self.FILEDIR + self.FILES[i]) 
            p, freqs, t, im = sf.spec(FILE)
            p = p
            sf.tellme('Select Song')
            plt.waitforbuttonpress()
            sf.tellme('Select Song')
            happy = False
            Proceed = False
            while not happy:
                happy = plt.waitforbuttonpress()

                if not happy:
                    tim = np.asarray(plt.ginput(n=0, timeout=-1))
                    ind = np.logical_and(t > tim[0,0], t < tim[1,0])
                    SongTime = mlab.find(ind)
                    
                    # Generate syllable spectrogram
                    self.winFreq = freqs > self.MINFREQ
                    self.winFreq = mlab.find(self.winFreq)
                    P = p[:,SongTime]
                    P = P[self.winFreq,:]
                    plt.close(1)
                    self.FIND_SONG = P.mean(0)
                    self.THRESHOLD = min(self.FIND_SONG) * 10.0
                    Proceed = True
                    break
                    
                if happy:
                    plt.close()
            if Proceed:
                break
        
    def cutSyllables(self, SYLLABLES, DISPLAYSYLL=False, DISPLAYSONG=False):
        """
        """
        
        if len(self.FILES) < self.MAXFILE + 1:
            ENDFILE = len(self.FILES)
        elif len(self.FILES) >= self.MAXFILE + 1:
            ENDFILE = self.MAXFILE
        
        song = 0
        fil = self.STARTFILE
        while fil < ENDFILE:
            FILE = (self.FILEDIR + self.FILES[fil])
            print 'processing file: ', fil
            ## Spectrogram ##       
            p, freqs, t, im = sf.spec(FILE)
            p = p * 0.9
            SongP, SongTime = sf.FindSong(p, self.FIND_SONG, t, self.winFreq,
                                         self.THRESHOLD)
            if SongP is False:
                plt.close(1)
                if len(self.FILES) < self.MAXFILE + 1:
                    ENDFILE = len(self.FILES)
                elif len(self.FILES) >= self.MAXFILE + 1 and ENDFILE < len(
                                                                self.FILES):
                    ENDFILE = ENDFILE + 1
                elif ENDFILE >= len(self.FILES):
                    ENDFILE = ENDFILE
                print 'Skipped because there was no song'
                pass
            
            ############# Find Syllables################
            elif SongP.any():
                mean_freq = np.zeros((len(SongTime),1))
                for j in range(0,len(SongTime)):        
                    fr = (sum(freqs[self.winFreq] * np.power(SongP[:, j], 2)) / 
                            sum(np.power(SongP[:, j], 2)))
                    mean_freq[j] = fr

                syllFound = sf.runs(SongP.mean(0) > self.THRESHOLD)
                FOUND = syllFound[0].any()
                if FOUND:
 
                    for syll in range(0,len(syllFound[0])):

                        time = np.arange(syllFound[0][syll],
                                         syllFound[1][syll])
                        if len(time) > 200:
                            plt.close(1)
                            plt.close(2)
                            print 'Skipped syllable because it was too long'
                            pass
                        elif len(time) <= 200:
                            P = SongP[:,time]
                            ent, mean_freq, duration = sf.measure(P, 
                                                                  time, 
                                                                  freqs, 
                                                                  self.winFreq, 
                                                                  t)
                            
                            if syll <= SYLLABLES - 1:

                                self.syllables[syll]['entropy'][song] = ent
                                self.syllables[syll]['freq'][song] = mean_freq
                                self.syllables[syll]['duration'][song] = (
                                                                    duration)
                                # Generate syllable spectrogram
                                if DISPLAYSYLL == 1:
                                    Z = 10*np.log10(abs(P))
                                    plt.figure()
                                    plt.imshow(Z,origin='lower', aspect='auto')
                                    plt.hold(True)
                                    plt.show(bloc=False)
                                    plt.tight_layout()
                                    plt.close(3)
                                plt.close(2)
                        plt.close(1)
                    song += 1
                                   
                if not FOUND:

                    plt.close(1)
                    plt.close(2)
                    if len(self.FILES) < self.MAXFILE + 1:
                        ENDFILE = len(self.FILES)
                    elif (len(self.FILES) >= self.MAXFILE + 1 and 
                            ENDFILE < len(self.FILES)):
                                
                        ENDFILE = ENDFILE + 1
                        
                    elif ENDFILE >= len(self.FILES):
                        ENDFILE = ENDFILE
                        print 'Skipped because there were no syllables \
                                extracted'           
            fil +=1

    def analysis(self):
        """
        """

        for syll in self.syllables:
            
            #Histograms: Entropy and Frequency
            #Entropy
            ent = []
            for song in self.syllables[syll]['entropy']:
                if song is not None:
                    
                    ent.append(self.syllables[syll]['entropy'][song])
            ent = np.array([num[0] for elem in ent for num in elem])
            #print ent
            DE_X = np.arange(-14,0.1,0.25)
            DAE,DAE_X = np.histogram(ent, bins=DE_X, density=False)
                                     
            self.syllables[syll]['dstEnt'] = DAE
            self.syllables[syll]['binsEnt'] = DAE_X
            
            #Frequency                      
            freq = []
            for song in self.syllables[syll]['freq']:
                if song is not None:
                    freq.append(self.syllables[syll]['freq'][song])
            freq = np.array([num[0] for elem in freq for num in elem])
            
            #FREQBIN = 100
            DF_X = np.arange(self.MINFREQ, np.max(freq), 100)
            DAF,DAF_X = np.histogram(freq, bins=DF_X, density=False)
                                     
            self.syllables[syll]['dstFreq'] = DAF
            self.syllables[syll]['binsFreq'] = DAF_X
                
                
            #Duration, mean
            dur = []
            for song in self.syllables[syll]['duration']:
                if song is not None:
                    dur.append(self.syllables[syll]['duration'][song])

            self.syllables[syll]['meanDur'] = np.mean(dur)
            self.syllables[syll]['stdDur'] = np.std(dur)
      
    def findKSstat(self):                       
        """
        """        
        # Load baseline files for comparison:
        baseline = self.loadPickle(condition='baseline')

        # KS stats:
        for syll in self.syllables:

            AED, AEp = stats.ks_2samp(self.syllables[syll]['dstFreq'], 
                                     baseline[syll]['dstFreq'])
            self.syllables[syll]['EntKS'] = AED
            self.syllables[syll]['EntPvalKS'] = AEp
            print 'syll ', syll, 'entropy : ', AED 

            AFD, AFp = stats.ks_2samp(self.syllables[syll]['dstEnt'], 
                                     baseline[syll]['dstEnt'])
            self.syllables[syll]['FreqKS'] = AFD
            self.syllables[syll]['FreqPvalKS'] = AFp
            print 'syll ', syll, 'freq : ', AED 
            
            EXPdur = []
            for song in self.syllables[syll]['duration']:
                if song is not None:
                    EXPdur.append(self.syllables[syll]['duration'][song])
                    
            BASEdur = []
            for song in baseline[syll]['duration']:
                if song is not None:
                    BASEdur.append(baseline[syll]['duration'][song])
                    
            ADT, ADp = stats.ks_2samp(EXPdur, BASEdur)
            self.syllables[syll]['DurKS'] = ADT
            self.syllables[syll]['DurPvalKS'] = ADp
            print 'syll ', syll, 'duration : ', AED            

    def distPlotter(self, fontsize=20):
        """
        """
 
        for syll in self.syllables:

            #Entropy
            plt.figure(figsize=(11,9))
            ax1 = plt.subplot(111)
            pf.AxisFormat()
            pf.TufteAxis(ax1, ['left','bottom'])
        
            ax1.bar(self.syllables[syll]['binsEnt'], 
                    (self.syllables[syll]['dstEnt'] / 
                                np.sum(self.syllables[syll]['dstEnt'])), 
                   width=0.2, color='r')
            ax1.set_ylabel('proportion', fontsize=fontsize)
            ax1.set_xlabel('entropy', fontsize=fontsize)
            ax1.set_xlim(-14,0)
            plt.title('syllable ' + str(syll + 1), fontsize= fontsize)
            plt.show()
         
            ax2 = plt.subplot(111)            
            pf.AxisFormat()
            pf.TufteAxis(ax2, ['left','bottom'])
            xcoords = self.syllables[syll]['binsFreq'][:]
            print 'len', len(xcoords), len(self.syllables[syll]['dstFreq']) 
            ax2.bar(xcoords, 
                    (self.syllables[syll]['dstFreq'] / 
                                np.sum(self.syllables[syll]['dstFreq'])), 
                    color='r', edgecolor=None)
            ax2.set_xlim(self.MINFREQ, 8000)
            #ax2.set_xticks(np.arange(0, 8001, 1000), 
            #               ['0','1','2','3','4','5','6','7','8'])
            ax2.set_ylabel('proportion', fontsize=fontsize)
            ax2.set_xlable('frequency')
            plt.title('syllable ' + str(syll), fontsize=fontsize)
            plt.tight_layout()
            plt.show()

    def durationPlotter(self, fontsize=20):
        """
        """
        for syll in self.syllables:
            
            plt.figure()
            plt.plot(syll, self.syllables[syll]['meanDur'],
                    yerr=self.syllables[syll]['stdDur'],
                    color='w', linewidth=3, align='center',
                    ecolor='k',capsize=10)
                    
            plt.title('Duration', fontsize= fontsize)
            plt.ylabel('syllable length (s)', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            #plt.xticks(np.arange(1,self.SYLLABLES+1), Lables, size='large')
            plt.xlabel('syllable ' + str(syll), fontsize=fontsize)
            
            plt.tight_layout()
            plt.show()

    def KSplotter(self):
        """
        """
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
        
    def euclidianPlotter(self):
        """
        """
        
        plt.figure()
        Magnitude = np.zeros((self.SYLLABLES, len(dur[0,:])))
        
        for j in range(0,SYLLABLES):
            for i in range(0,len(dur[0,:])):
        
                d = np.sqrt( ((dur[j,i]-0)**2) + ((entropy[j,i]-0)**2) + 
                            ((freq[j,i]-0)**2))
        
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


    def meanEuclidianPlotter(self):
        """
        """

        MEANCOLOR = 'b'
        
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
        plt.fill_between(np.arange(0,len(meanMag[0,:])),meanMag[0,:], 
                         meanMag[0,:]+stdvMag[0,:],
                         facecolor=MEANCOLOR,alpha=0.5)
        plt.fill_between(np.arange(0,len(meanMag[0,:])),meanMag[0,:], 
                         meanMag[0,:]-stdvMag[0,:],
                         facecolor=MEANCOLOR,alpha=0.5)
        plt.xticks(np.arange(0,len(dur[0,:])),DPS,size=20)
        plt.yticks(size=20)
        plt.tight_layout()
        
        plt.show()
        plt.savefig(BIRDNAME+'/'+BIRDNAME + '_MeanMag.png')
            
    def loadPickle(self, name=None, date=None, condition=None):
        """
        """
        from pickle import load
        
        if not name:
            name = self.BIRDNAME
        if not date:
            date = self.DATE
        if not condition:
            condition = self.CONDITION 
        
        if condition == 'baseline':
            f = open(('analysis/' + name + '/' + name + 'baseline' +
                    '.pickle'), 'r')
        else:
            f = open(('analysis/' + name + '/' + name + 'experiment' + date + 
                    '.pickle'), 'r')        
        return load(f)
            
    def savePickle(self, meta=None):
        """
        """
        from pickle import dump
        
        if not meta:
            if self.CONDITION == 'baseline':
                f = open(('analysis/' + self.BIRDNAME + '/' + self.BIRDNAME + 
                    self.CONDITION + '.pickle'), 'w')
            else:
            
                f = open(('analysis/' + self.BIRDNAME + '/' + self.BIRDNAME + 
                    self.CONDITION + self.DATE + '.pickle'), 'w')
            dump(self.syllables, f)

        if meta:
            f = open(('analysis/' + self.BIRDNAME + '/' + self.BIRDNAME + 
                        'metadata.pickle'), 'w')
            dump(self.syllables, f)
            
    def returnSyllables(self):
        """
        """
        return self.syllables
    
if __name__ == '__main__':
    
    #os.chdir('/Users/brianschmidt/Projects/birdsong/')
    
    # BIRD INFO #
    BIRDNAME = 'bird_1958'
    CONDITION = 0 ## 0 = Baseline, 1 = Post Manipulation
    ANALYSIS = 1 ## Automatically compare Exp Condition to Baseline? 0 = No, \
                #1 = Yes
    DATE = 'Feb_2' ## Not important if Condition is 0
    SYLLABLES = 5 ## Number of syllables to be analyzed
    
    # ANALYSIS OPTIONS #
    MAXFILE = 10 ## Maximum number of songs to be analyzed
    
    if CONDITION == 0:
        FILEDIR = (BIRDNAME+'/baseline song/')
    elif CONDITION == 1:
        FILEDIR = (BIRDNAME+'/post-surgery song/'+ DATE +'/')
    
    STARTFILE = 0
    
    handle = SongAnalysis(SYLLABLES, FILEDIR, BIRDNAME, CONDITION, DATE,
                          MAXFILE)
    data = handle.returnSyllables()