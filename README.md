# GBK-means Clustering Algorithm
Welcome. This repository contains the matlab based implementation of the 'GBK-means clustering algorithm: An improvement to the K-means algorithm based on the bargaining game'. In this repository, source codes of GBK-means Clustering Algorithm and its comparisons with two well-knowed clustering algorithms, K-means and fuzzy cmeans, are presented. Comparisons have been made on artificial and real world data sets with regard to Common validity indexes. Proposed approach is a new mechanism for addressing cluster analysis problem in wich cluster centers compete with each other to attract the largest number of similar objects or entities to their cluster. 

# GBKmeans

Output = GBKmeans(X, Ncluster, PSOparams)
% This function is the implementation of the GBK-means Clustering
% Algorithm.


# Notes
In the our bargaining game based k-means clustering method, we are used a linear combination of the max and mean of inter cluster distances as a utility function for cenetr of clusters (players). Based on our best knowledge, in the different datasets with their embeded informations, different weight to each of the max and mean in the linear combination results to different accuracies. So, this matter may be addresseed by authers using to incorporate information theory based methods. 

# Citation
If you find this code useful please cite us in your work:
Mustafa Jahangoshai Rezaee, Milad Eshkevari, Morteza Saberi, Omar Hussain,
GBK-means clustering algorithm: An improvement to the K-means algorithm based on the bargaining game,
Knowledge-Based Systems,
Volume 213,
2021,
106672,
ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2020.106672.
(http://www.sciencedirect.com/science/article/pii/S0950705120308017)


