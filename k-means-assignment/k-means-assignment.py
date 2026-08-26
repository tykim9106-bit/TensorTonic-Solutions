def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    assignments = []

    for point in points:
        best_index = 0
        best_distance = float("inf")

        for i, centroid in enumerate(centroids):
            distance_square = 0

            for d in range(len(point)):
                distance_square += (point[d] - centroid[d]) ** 2

            if distance_square < best_distance:
                best_distance = distance_square
                best_index = i

        assignments.append(best_index)

    return assignments