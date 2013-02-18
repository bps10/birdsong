#|  birdsong
*A simple python program for analysis of White-crowned Sparrow song.*

This software has been designed to analyze bird song for the lab of Eliot Brenowitz. It has been specifically built to process White-crowned Sparrows and has not been tested with the song of other species.  The extraction algorithm employed here is a relatively crude heuristic that takes advantage of the highly stereotyped nature of White-crowned song. As such, it will necessarily fail with species, such as zebra finch, that do not sing syllables in a regular order.  

##| introduction

Birdsong has been developed in the [python](http://python.org) programming language.  Python is a cross platform, open source, interpretive language with an emphasis on readable syntax.  These attributes make it an attractive language for scientific computing.  The current program takes advantage of numerous open source projects built by the scientific community: the [numpy](http://numpy.org) (i.e., numerical python), [scipy](http://scipy.org)(i.e., scientific python), [matplotlib](http://matplotlib.org) and [guidata](http://code.google.com/p/guidata/) libraries.  All of these packages are necessary to run the program from source code.  The simplest way to install of these dependancies is through the [pythonXY](http://code.google.com/p/pythonxy/) distribution.  The [enthought suite](http://enthought.com) will work as well, though guidata will have to be installed separately and itself has numerous dependancies (making it a headache for someone unfamiliar with managing python libraries).  In the (ideally) not so distant future this installation process will not be necessary as the files will be compiled into an executable for easier distribution.

##| installation

To download from the command line, use:

`git clone https://github.com/bps10/birdsong`

Alternatively, you can download as a zip from the link at the top of this github [repo](http://github.com/bps10/birdsong).  After retrieving the source files, call the user interface with:

`python gui.pyq`

A proper setup.py will be written shortly for building the application and a distributable executable file should follow.

##| usage

The analysis has been designed to compare song from a baseline condition with song after experimental manipulation.  

*More to follow*

##| instructions

#### User interface:
1. Name the bird.  For now, do not leave any spaces - use underscores if necessary.
2. Indicate the number of syllables that the syllable parser should expect.  Typically these birds sing 5 syllables, which is the default value.
3. Enter the date the songs were collected (month/day/year; two digits each).
4. Select the directory from which to draw the songs.  This directory should include all of the songs sung from a single day (or a group of days if you want to set up the analysis that way).  The program will draw the songs in alphanumerical order. Thus, with the naming convention typically used in the Brenowitz lab, this will go from the earliest song of the day to the last song.  Keep this ordering in mind, as a birds song might change slightly during a day and therefore you may want to randomize this process.
5. Indicate the number of songs to process.  Default is 60.  This may be overkill in certain cases.
6. Set the minimum frequency.  Low frequencies are often corrupted with noise and well below the frequency range of the song, so we impose a simple high pass filter here to eliminate that noise.  The default is 1250Hz.
7. Choose the file to begin analysis with.  (See #5 for a brief description of why you might want to change this).  Usually, you won't need to alter this parameter.

#### Select song:

After filling in all of the parameters in the user interface, the program will launch.  You will be shown the first song in the stack and asked to 'select the song'.  The purpose here is to find a good song for the program to use as an example.  The approximate duration and mean frequency will be used for finding subsequent songs and avoiding random noises that were picked up by the mic. Use the forward arrow key to find a good example of a typical song from the bird.  When you have found one that you like use the mouse to click on the spectrogram.  You will have to click a total of 4 times.  

The first click one tells the program that you are going to use this song.  Nothing will happen.  The second click tells the program that you are going to select the start of the song and again nothing will happen.  The third click should occur at the beginning of the first syllable (actually about a half second before).  You should see a red cross hair show up where you clicked.  Make sure that you are not clicking too rapidly or you will place multiple cross hairs here and the program will not like it.  If you need to redo your selection you should be able to remove the cross hair with the delete key, but this does not always work.  Simply exit and start the program over again if you run into trouble.  Finally, click on the spectrogram one last time to indicate the end of the song, again leaving a little space from the actual end.  (The actual duration of a given song is slightly variable, so you want the program to have some working room.)  Once you are satisfied, hit enter and the program will begin running.  At this point consider looking away from the screen as all of those flashing spectrograms can be a bit nauseating! (Eventually we will add an option to suppress this behavior, but for now it is useful to ensure that nothing terribly wrong is happening.)

After completion of the song extraction and analysis, the program will run some statistics and show you a bunch of summary plots.  You can save them if you would like or simply close out of them.  


###| data structures

Currently, the data are saved in the analysis folder into separate folders for each bird.  The data will be written into pickle files for now, which is a python data structure designed to preserve the object structure of pythonic data.  They are analogous to matlabs .mat files.  A method for dumping this data into csv files is on the way.

Within the main class the data are organized in a syllable centric manner…

###| analysis

*details of the analysis will be written soon*

###| source documentation

*coming soon*



