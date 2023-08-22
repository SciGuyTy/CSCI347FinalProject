# Description
As datasets continue to grow in size and complexity, the use of traditional supervised learning techniques for large-scale research efforts is becoming increasingly expensive and time consuming. To address these challenges, many projects have begun to look towards unsupervised learning methods which are better equipped for unlabeled and complex data. Among these methods, K-Means clustering has perhaps been the most well-adopted, largely due to its simplicity, flexibility, and performance. However, the default K-Means algorithm is not without its limitations. Most notably, the random initialization of cluster centroids exposes the algorithm to getting caught in local, non-optimal minima. This can severely harm clustering performance, especially with highly-clustered or high-dimensional data (where clusters may be very close together). Our research attempts to better understand the impact of this problem and how it might be addressed by comparing the performance between K-Means and two alternative implementations: K-Means++ and the Hartigan Wong K-Means algorithm. 

_Taken from CSCI 347 Final Project Report_

# Repository Structure
The repository is broken up into four sections: _algorithms_, _data_, _utilities_, and the _experiment_. 
- Algorithms: This contains an implementation of the base **K-Means** algorithm as well as our implementation of the three initialization methods under consideration
- Data: This contains the Original Breast Cancer Wisconsin dataset, sourced from the UCI Machine Learning Repository
- Utilities: This directory contains a set of reusable utility classes that abstract common and complex functionality used throughout our experiments
- Experiment: This Jupyer notebook contains the implementation of our experiment as documented in the _CSCI 347 Final Project Report_
