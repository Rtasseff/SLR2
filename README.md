SLR2 - Tools for Sparse Linear Regression second version 
==================================
20111212 RAT

These are tools and scripts to run various analysis workflows 
using sparse linear regression.  
This is a second version compared to SLRplus, which was avery general set of tools.  
SLR2 is far more specific for workflows that I have tested and some I have found valuable.
These workflows center around creating netwroks of coupled oscillators
where the couplings (or network edges) are defined as the non zero 
regression coefficents; however, these tools can be used for any elastic net regression.


Functionality
-------------
Lots of stuff going on in here.
Tools to estimate p-values, improtance scores 
tools for choosing parameters, calculating validation metrics,
estimating errors and so on.
Many of the methods are typcially time intensive, some simplifications are avalible,
but the accuracy/differences are not tested (I think most of the simplificaitons give poor results) 
The most valuable found for identifying network structure (not as much the actual values)
can be found in SLR2\_3.py called run\_1, the output is set up so that jobs can be distributed 
for many regressions and the results aggrigated into a single set of files.
 

Depaendencies
------------
I used python 2.7, numpy 1.6.1, and scipy 0.10.0.

IMPORTANT
You need the wrapper and the original fortran code for glmnet to do anything useful.
The code is maintained at: https://github.com/dwf/glmnet-python.
There is a little trick in the compile step so read the readme for that module.

To improve efficiency of p-value calculations I used an analysis (and code) from:
http://informatics.systemsbiology.net/EPEPT/.  The original code was in matlab,
and I converted it to python around 20111205. (https://github.com/Rtasseff/gpdPerm)

To allow for multiprocessing of jobs I used code (dispatcher/) from a colleague (JR) 
which may be included, this only impacts the files runSLR.py and runMP.py and not
the above functionality. (https://github.com/Rtasseff/dispatch)



License
-------

There is a license for the wrapper and the original fortran code but it is not
my intention to distribute or maintain that code so I will leave you to find it 
when and if you decide to download it (note: it is freely available).

Copyright (C) 2003-2013 Institute for Systems Biology, Seattle, Washington, USA.
 
The Institute for Systems Biology and the authors make no representation about the suitability or accuracy of this software for any purpose, and makes no warranties, either express or implied, including merchantability and fitness for a particular purpose or that the use of this software will not infringe any third party patents, copyrights, trademarks, or other rights. The software is provided "as is". The Institute for Systems Biology and the authors disclaim any liability stemming from the use of this software. This software is provided to enhance knowledge and encourage progress in the scientific community. 
 
This is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
http://www.gnu.org/licenses/lgpl.html
