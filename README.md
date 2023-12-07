# tokamak-unsupervised

What we did:
PCA 3D of a few shots
Saw that the data was "continuous"
Validated this using DBSCAN: shots were put into one cluster

What we need to do:
Use UMAP (clustering-algo) on as many shots as possible and look for insights
Decide how to consider time
We re going to do some unsupervised clustering and PCA, but can do supervised stuff to consider corelations between shots and machine inputs.

The objective of the project is to get a deeper understanding of what machine inputs result in the QCEH states that we are interested in.
Many shots have small differences in parameters as they build upon eachother. The researchers are quite biased as they read papers and have biased opinions about what results in better experiments. Our job is to find patterns and try to back up or even refute these assumptions.



We also disccused the following thing but its not our priority for now
Decided to do some kind of KNN in lower PCA space