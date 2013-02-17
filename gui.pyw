# -*- coding: utf-8 -*-
import guidata
_app = guidata.qapplication() 
# not required if a QApplication has already been created

import guidata.dataset.datatypes as dt
import guidata.dataset.dataitems as di

## import birdsong
from birdsong import SongAnalysis

class Processing(dt.DataSet):
    """Run birdsong"""    
    birdname = di.StringItem("name of the bird")
    syllables = di.IntItem("Number of syllables: ", min=1, max=10, default=5)
    condition = di.ChoiceItem("Condition:", ("baseline", "experimental"))
    date = di.DateItem("date collected")
    directory = di.DirectoryItem("file directory")
    maxfiles = di.IntItem("Max # of file to process", default=60)
    minfreq = di.IntItem("minimum frequency", default=1250)
    startfile = di.IntItem("start file", default=0)
         
param = Processing()
param.edit()
print str(param.date).replace('-','_')

SongAnalysis(param.syllables, param.directory, param.birdname, param.condition,
             str(param.date).replace('-','_'), param.maxfiles)