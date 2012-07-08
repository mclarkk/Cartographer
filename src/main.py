# Author: Meghan Clark
# Date: 6/29/12
# Description: The main method for Cartographer.

# Get batch parameters
# These should come from a file at some point.
dimension = 2
point_counts = [10, 100]
core_counts = [5, 10]
mappings = ['hilbert', 'zorder']
k_nearest_neighbor_counts = [2, 3]
nearness_neighborhoods = [1, 2]

# Data structures
# Points are actually indices
coordinates #[]# points
distance_matrix #[][] # points x points
nearest_neighbors #[][] # points x knncs
near_neighbors #[][] # points x nns
avg_dist_to_nearest_neighbors #[][] # points x knncs
points_1D #[][] # points x mappings
m_knn_ordering #[][][] # points x mappings x knncs
m_nn_ordering #[][][] # points x mappings x nns
m_knn_cores #[][][][] # points x mappings x knncs x cc
m_nn_cores #[][][][] # points x mappings x nns x cc


# Run experiment batch.

# sort point counts.

# LOOP: for each point count
#   generate point_count points (or add to existing)
#   construct distance matrix
#   LOOP: for each nearness_range
#     Get ordered list of near neighbors
#   END
#   LOOP: for each nearest_neighbor_count
#     Get ordered list of nearest neighbors
#     metric: get avg. distance to nearest neighbors,
#   END
#   LOOP: for each mapping
#     For each point, get 1-D image
#     Sort in order
#     LOOP: for each nearest_neighbor counts
#       metric: % of k nns within k steps in 1D ordering
#     END
#     LOOP: for each nearness range
#       metric: % of near ns within <# near ns> steps in 1D ordering
#     END
#     LOOP: for each core count
#       Divide ordering into core_count chunks
#       LOOP: for each nearest_neighbor counts
#         metric: % of knns that end up on the same core
#       END
#       LOOP: for each nearness range
#         metric: % of near ns that end up on the same core
#       END
#     END
#   END
# END



# Dump data to file.
# Or in this case, print

def printf_array(dimensions, labels, array):
	# Make sure dimensions matches # labels
	
