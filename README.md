# User guide
## Build guide
The Mahalanobis-average hierarchical clustering project was developed with the
CMake build tool. 

To build the executable, use CMake configure and build commands in a build directory. Then, the directory `para` will contain gmhclust executable. 

The only dependency is the CUDA compiler (nvcc). The executable
should be portable to all platforms supporting nvcc; it was successfully tested
on Ubuntu 18.04 and Windows 10. 

See the following steps:
```
cd gmhc
mkdir build && cd build
cmake ..
cmake --build .
ls para/gmhclust
```
## Running the program
The gmhclust executable has three command line parameters:

1. *Dataset file path* – The mandatory parameter with a path to a dataset file.  
The file is binary and has structure as follows:  
A. 4B unsigned integer *D* – point *dimension*  
B. 4B unsigned integer *N* – *number* of points  
C. *N*.*D* 4B floats – *N* single-precision *D*-dimensional points stored one
after another

2. *Mahalanobis threshold* – An absolute positive number that states the Mahalanobis threshold. It is the mandatory parameter.

3. *Apriori assignments file path* – An optional path to an apriori assignments
file — a file with space separated 4B unsigned integers (assignment numbers). The number of integers is the same as the number of points in the
dataset; it sequentially assigns each point in the dataset file an assignment
number. Then simply, if the *i*-th and the *j*-th assignment numbers are
equal, then the *i*-th and *j*-th points are assigned the same apriori cluster.


The executable writes the clustering process to the standard output. Each
line contains an ID pair of merged clusters with their merge distance as well.  
The command, that executes the program `gmhclust` to cluster `data` dataset
with the apriori assignment file `asgns` and the threshold 100 is  
```./gmhclust data 100 asgns```
