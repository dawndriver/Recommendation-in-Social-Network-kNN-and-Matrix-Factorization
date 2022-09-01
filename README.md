# Recommendation-in-Social-Network-kNN-and-Matrix-Factorization

In this project, we use the problem from KDD Cup 2012 (http://www.kddcup2012.org/c/kddcup2012-track1/data) to predict if a user will follow an item that has been recommended to the user (items can be persons, organizations, or groups). With the data provided, a model will be used to predict if a user will follow certain items.

In this project, Semi-Lazy Mining Paradigm (LAMP) is applied to first find k nearest neighbors (kNN) upon query and then build a mini-model of those neighbors to do prediction. In the kNN stage, we use MinHash and LSH technique to quickly find similar users upon query. Then a mini-utility-matrix with those neighbors is built and Matrix Factorization technique is applied to get the approximate (non- known) interest of users to items.
