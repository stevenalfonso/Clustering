using Clustering
using Distances

# Generate sample data
data = rand(2, 100) * 10

# Define the parameters for DBSCAN
ϵ = 1.0
minpts = 5

# Compute the pairwise distance matrix
distances = pairwise(Euclidean(), data, data)

# Perform DBSCAN clustering
clusters = dbscan(distances, ϵ, minpts)

# Print the results
for (i, cluster) in enumerate(clusters)
    println("Cluster $i: ", cluster)
end
