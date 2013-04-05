#!/bin/sh

cd value_functions
mcc -W lib:value2 -T link:lib value2.m groups3.m DiffManhattanDist.m libmmfile.mlib 
#-a ../Training0202.mat
mv value2.so libvalue2.so
cd ..
