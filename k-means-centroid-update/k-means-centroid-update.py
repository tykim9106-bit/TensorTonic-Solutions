def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Write code here
    centroids = []

    for i in range(k):
        cluster_points = []

        for index, point in enumerate(points):
            if assignments[index] == i:
                cluster_points.append(point)

        centroid = []
        for d in range(len(points[0])): 
            sum = 0
            for point in cluster_points:
                sum += point[d]
            if len(cluster_points) == 0:
                centroid.append(0)
            else:
                centroid.append(sum/len(cluster_points))
        centroids.append(centroid)

    return centroids