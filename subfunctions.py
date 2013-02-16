# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
import wave as wv


def runs(bits):
    """
    """
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

def measure(sample, time, freqs, winFreq, t):
    """
    """
    # Create matrix
    W = np.zeros((len(time),1))
    mean_freq = np.zeros((len(time),1))
    # calculate Weiner Entropy and Mean Freq
    for j in range(0,len(time)):
        w = np.log((np.exp(sum(np.log(sample[:,j])) / len(sample[:,j])))/
                    (sum(sample[:,j]) / len(sample[:,j])))
        fr = sum(freqs[winFreq]*np.power(sample[:,j],2))/sum(np.power(sample[:,j],2))
        W[j]= w
        mean_freq[j] = fr
    # Duration - in seconds
    duration = max(t[time])-min(t[time])
    return W, mean_freq, duration

def spec(FILE):
    """
    """
    spf = wv.open(FILE,'r')
    sound_info = spf.readframes(-1)
    sound_info = plb.fromstring(sound_info, 'Int16')
    f = spf.getframerate()
    plt.figure(num=None, figsize=(16, 7), dpi=80, edgecolor='k')
    p, freqs, t, im = plb.specgram(sound_info, Fs = f, scale_by_freq=True,
                                   sides='default',)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    font = {'size'  : 18}
    plt.rc('font', **font)
    plt.bbox_inches='tight'
    #plt.tight_layout()
    plt.show(block=False)
    return p, freqs, t, im


def FindSong(p,q,t, winFreq, THRESHOLD, SONGPERCENT=0.5, DISPLAYSONG=0):
    """
    """
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

    
def tellme(s):
    """
    """
    print s
    plt.title(s,fontsize=16)
    plt.draw()