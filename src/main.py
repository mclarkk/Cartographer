# Author: Meghan Clark
# Date: 6/29/12
# Description: The main method for Cartographer.

import random
import math
#import pprint
from scurve import hilbert #hilbert_point, hilbert_index
from scurve import zorder #

# Get batch parameters
# These should come from a file at some point.
hypercube_bound = 50.0 #This shouldn't have an effect on outcome, but what if it does?
order = 8 #Does this affect the outcome? How should I determine this?
dimension = 2
point_counts = [5] # [10, 100] #Point counts should be higher than core counts
core_counts = [5, 10]
mappings = ['hilbert', 'zorder']
k_nearest_neighbor_counts = [2, 3]
nearness_neighborhoods = [1, 2]
verbosity = 2

# Data structures
# Points are actually indices
coordinates = [] #points
discretized_coords = [] #dynamically drawn grid squares containing points. This is also how we'll determine the curve order to use. Looks like it's quadtree time after all. Sigh.
#distance_matrix #[][] # points x points
#nearest_neighbors #[][] # points x knncs
#near_neighbors #[][] # points x nns
#avg_dist_to_nearest_neighbors #[][] # points x knncs
#points_1D #[][] # points x mappings
#m_knn_ordering #[][][] # points x mappings x knncs
#m_nn_ordering #[][][] # points x mappings x nns
#m_knn_cores #[][][][] # points x mappings x knncs x cc
#m_nn_cores #[][][][] # points x mappings x nns x cc


# Run experiment batch.

def main():
	if verbosity > 0:
		print_parameters()

	# sort point counts.
	point_counts.sort()
	# sort nearness ranges and knn counts
	k_nearest_neighbor_counts.sort()
	nearness_neighborhoods.sort()
	max_pc = point_counts[-1]
	max_knnc = k_nearest_neighbor_counts[-1]
	# initialize distance matrix
	distance_matrix = [[0 for j in xrange(max_pc)] for i in xrange(max_pc)]
	# initialize nearest neighbors
	nearest_neighbors = [[[0 for k in xrange(len(k_nearest_neighbor_counts))] for j in xrange(len(max_pc))] for i in xrange(len(point_counts))]
	# initialize near neighbors
	near_neighbors = [[[0 for k in xrange(len(nearness_neighborhoods))] for j in xrange(len(max_pc))] for i in xrange(len(point_counts))]

	# LOOP: for each point count
	for point_count in point_counts:
	# generate point_count points (or add to existing), and fill in distance matrix as you go along.
		for i in range(point_count):
			xcoord = random.uniform(-hypercube_bound, hypercube_bound) #try diff distributions? 
			ycoord = random.uniform(-hypercube_bound, hypercube_bound) 
			coordinates.append((xcoord, ycoord))
			# get the discretized coordinate. Will write algorithm later, hack for now.
			discretized_coords.append((int(round(xcoord)), int(round(ycoord))))
			# fill out distance matrix as you go
			for j in range(i):
				dist = euclidean_dist(coordinates[i], coordinates[j])
				# take advantage of the symmetry
				distance_matrix[i][j] = dist 
				distance_matrix[j][i] = dist
			distance_matrix[i][i] = 0
		if verbosity > 0:
			print "Coordinates:"
			for c in enumerate(coordinates):
				print c
		if verbosity > 1:
			print "\nDistance matrix:"
			printf_array(2, [""], distance_matrix) # TEST
			print

		for i in range(point_count):
			dist_to = distance_matrix[i] #selects row of matrix
			# Get lists of near neighbors
			for radius in nearness_ranges:
				near_neighbors[pc_index][i][indexof(radius)] = [p for p in range(point_count) if dist_to[p] <= radius]
			# Get ordered list of nearest neighbors
			
		
		#   metric: get avg. distance to nearest neighbors,
		# END
		# LOOP: for each mapping
		for map in mappings:
			if map == 'hilbert':
				# For each point, get 1D image
				ordering_1D = [(hilbert.hilbert_index(dimension, order, c[1]), c[0]) for c in enumerate(discretized_coords)]
				# Sort into 1D order by hilbert distance (called hilbert_index above). 
				# Each element is of form (hilbert_distance, coordinate_index) 
				ordering_1D.sort()
				if verbosity > 0:
					print "Hilbert ordering (Hilbert distance, particle #):\n", ordering_1D, "\n"
			else:
				print "Map '{0}' not supported.".format(map)

	#     LOOP: for each nearest_neighbor counts
	#       metric: % of k nns within k steps in 1D ordering
	#     END
	#     LOOP: for each nearness range
	#       metric: % of near ns within <# near ns> steps in 1D ordering
	#     END
	#     LOOP: for each core count
			for core_count in core_counts:
				# Divide ordering into core_count chunks
				chunk_size = len(ordering_1D)/core_count
	#       LOOP: for each nearest_neighbor counts
	#         metric: % of knns that end up on the same core
	#       END
	#       LOOP: for each nearness range
	#         metric: % of near ns that end up on the same core
	#       END
	#     END
	#   END
	# END



# Dump data to file. Or print, whatever. Should probably have a verbosity flag.

# It is assumed that all points have same number of dimensions.
# Euclidean dist between n-dimensional points a and b: 
#   sqrt((a0-b0)^2 + (a1-b1)^2 + ... + (an-bn)^2)
def euclidean_dist(a, b):
    return math.sqrt(math.fsum([math.pow(a[i]-b[i], 2) for i in range(len(a))]))
    
def printf_array(dimensions, labels, array):
	# Make sure dimensions matches # labels
	if dimensions == 2:
		for i in range(len(array)):
			for j in range(len(array[i])):
				print '{0:10.3f}'.format(array[i][j]),
			print

def print_parameters():
	print """Report verbosity: {8}\n
************ PARAMETERS ************
Hypercube boundaries: {0}
Hypercube dimension: {1}
Curves: {2}
Curve order: {3}
Point counts: {4}
Core counts: {5}
K-Nearest-Neighbor counts: {6}
Nearness neighborhoods: {7}
************************************\n""".format(hypercube_bound, dimension, mappings, order, point_counts, core_counts, k_nearest_neighbor_counts, nearness_neighborhoods, verbosity)

if __name__ == "__main__":
  main()
