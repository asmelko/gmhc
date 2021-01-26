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


The command, that executes the program `gmhclust` to cluster `data` dataset
with the apriori assignment file `asgns` and the threshold 100 is  
```./gmhclust data 100 asgns```

### Output

The executable writes the clustering process to the standard output in a text format. Each
line contains an ID pair of merged clusters with their merge distance as well.  
IDs are assigned as follows:
1. Initial dataset points are assigned nonnegative integers (`[0, n-1]`).
2. Merged clusters are assigned the next possible ID (`[n, 2n-1]`).  

An example output for 4 points in a dataset would look like this:
```
0 2 0.65
1 4 1.2
3 5 0.1
```
## R package build guide

To build the package, use CMake configure and build commands in a build directory. Specifically, build target `gmhc_package`.

See the following steps (last step to work, you need to have root rights):
```
cd gmhc
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target gmhc_package
```
Building the specified target installs `gmhc` package to the default R package directory. 
Then in R session, it can be used as follows:
```
library('gmhc')
?gmhclust
```