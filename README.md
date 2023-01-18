# NESMOTE (WIP)
A collection of augmetation methods applyable in non-euclidean spaces. All classes are designed using the samebasic pattern which allows for quick switching between different solutions and easy integration.

## prepreqisites
Algorithms require user to define metrics in the desired space which refet to points as rows inside numpy array or pandas dataframe as well as weighted average of points that will be used to generate synthetic data.

## testing
Testing is performed on regular euclidean data and the goal is to achieve reasonable performance while maintaining high accuracy. 

So far the accuracy requirements are met completely and the possible improvements are being implemented and tested/

## future
The C++ version will be implemented after the results become satisfactory enough to call the project a success.