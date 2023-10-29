<!-- <span style="color:red">Your red text here</span> -->

# Supervised Learning I

* Logistic regression
    * Classification, the output is log odds
    * What is the decision boundary? 
* Resampling methods
    * Cross-validation
        * Why do we need it? Test error rate
        * Drawbacks? Not trained on full data, the test error rate depends on which section 
        * 

# Supervised Learning II

* SVM
    * How can SVM output probabilities? Fit a logistic regression model to function score - Platt scaling
* Decision tree
    * Greedy recursive binary splitting
        * At every step, select a feature and threshold for splitting that decreases RSS the greatest
    * Suffers from overfitting if not stopped
    * Keep fitting until the decrease in RSS exceeds some threshold. The problem is that there might be a good fit later on.
    * Cost complexity pruning/weak link pruning
        * Regularize with the number of terminal nodes. Choose the hyperparameter alpha through cross-validation
        * Build the full tree until the observations at the leaf are less by a threshold
        * Vary alpha from a range of values. Keep pruning the weakest link based on RSS until minimum RSS with regularization is achieved 
    * For classification use Cross entropy, 
        * Classification error: Fraction of observations that do not belong to the most occurring class. However, this is not sensitive to tree growing as it doesn’t consider other classes
        * A node is pure if it contains observations from only one class
            * Gini index: sum of p_k(1-p_k) p_k being the fraction of observations of a particular class k
            * Cross entropy: -p_k*log(p_k)
            * Both are actually quite similar. Expansion of log(1/x) is 1-x
* Bagging ensemble
    * <span style="color:red">What is bagging and why do we do it?</span>
    * <span style="color:red">What is out-the-bag score?</span>
* Random Forest Classifier
    * <span style="color:red">What are max_samples and what does bootstrap mean?</span>
* Boosting
    * <span style="color:red">What is the problem it solves compared to a single decision tree and random forest?</span>
    * <span style="color:red">Difference between AdaBoost and XGBoost?</span>
* Ensemble methods
    * Hard voting - just take max
    * Soft voting - average the probabilities

# Unsupervised Learning

* Dimensionality Reduction 
    * Easier/faster to train and easier to visualize
    * Linear - PCA
    * Nonlinear - Isomap, tSNE
* PCA
    * Imagine data around the y=x line where points are more spread along the line
    * Center the data around the mean. 
    * Finds dimensions along which the variance is high
    * Method 1: Optimization method - iteratively find the most significant components
        * Optimize for weights of the first component that maximizes the variance
        * Optimize for weights of the second component that maximizes the remaining variance. <span style="color:red">How to calculate the remaining variance?
    * Method 2: Decompose to find the top eigenvectors of sample covariance
        * <span style="color:red">Why do eigenvectors give principal components?
        * 
* Isomap
    * Imagine a Swiss roll in 3D. 
    * The first component would be a non-linear one that lets the data spread.
    *  PCA can only give linear ones, so it will not be an ideal here
    * Isomap preserves the geodesic distances 
    * Use a multidimensional scaling algorithm - if you know how far the points are in terms of similarity, we can map them to a lower dimensional map that preserves their similarity. 
    * <span style="color:red">How does the algorithm exactly work? Can we map new points?
* tSNE (t-distributed stochastic neighbor embedding)
    * Mainly used for visualization. High dimensional data to two/three dimensions. Preserving local structures and revealing global structures
    * Finds similarity between two points using Gaussian distribution in SNE, t distribution in tSNE. The closer points will have a higher probability and the farther ones will have a lower probability. Construct a matrix of pairwise probabilities. 
    * Construct a similar matrix in lower dimensions 
    * Reduce the distance (KL divergence) between these probability matrices using gradient descent
    * T-distribution has heavier tails - <span style="color:red">which solves the crowding problem
    * Perplexity - parameter that controls the effective number of neighbors
* Clustering
    * K means
        * Cluster points into k clusters
        * Start with random k cluster centroids, find the clusters, find the centroids of the new clusters, and keep doing so until the cost function - distance b/w cluster center and point doesn’t decrease much
        * We can choose the same criteria for choosing the hyper parameter k
    * Fuzzy K means
        * K means does hard clustering, fuzzy k means assigns degree of membership to each point
        * Many ways to calculate the degree of membership
            * Gaussian distance. Sigma controls the fuzziness
            * Eucledian distance but a way to control the fuzziness
    * Hierarchical clustering - agglomerative
        * Start with each point as a cluster, merge the two clusters that are closest to each other, continue doing until you have only one cluster
        * Various distance metrics across clusters can be used  - avg distance, nearest distance, farthest distance, centroid distance
    * Hierarchical clustering - divisive
        * Start with one big cluster, split it so that the resulting two clusters has the least distance. 
    * We don’t need to know K beforehand for hierarchical clustering
    * Both hierarchical clustering algos are irreversible
    * <span style="color:red">How does hierarchy differ from Kmeans?
* Collaborative filtering
    * Netflix: users rate movies. How do we recommend movies to users?</span>
    * Find top K movies to recommend to users. 
    * Basic algorithm
        * The similarity score between the two movies. Represent each movie as a vector of ratings by users. 
        * How to handle missing data - mean centering (movie that is not rated gets 0)
        * Predict the rating of every other movie by taking the weighted average of ratings of K movies that the user rated
    * SVD
        * Decompose a matrix into U\SigmaV^T called singular value decomposition - Decompose into latent features like genres
        * Choose top k singular values and reconstruct the matrix U_k\Sigma_kV_k^T
        * rating_ij = corresponding element in the reconstructed matrix
        * <span style="color:red">How is SVD robust to missing values?</span>
    * <span style="color:red">SVD++, Factorization machines