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
core_counts = [2]
mappings = ['hilbert', 'zorder']
k_values = [2, 3] #for k nearest neighbor metrics
nearness_radii = [30, 50] #for near neighbor metrics
verbosity = 5

# DATA STRUCTURES
# Coordinates for each point. Points are indices (implicitly).
coordinates = []
# Dynamically drawn grid squares containing points. This is also how we'll 
# determine the curve order to use. Looks like it's quadtree time after all. 
# Sigh. Hack it for now.
discretized_coords = [] 
#distance_matrix = [][]
nearest_neighbors = {} #keys: (pc, point, knnc)
near_neighbors = {} #keys: (pc, point, dist)
#avg_dist_to_nearest_neighbors #[][] # points x knncs
#knn_conservation_1D #[][][] # points x mappings x knncs
#knn_conservation_1D_stats = {} #(point count, mapping, k value)
nn_conservation_1D = {} #[][][] # points x mappings x nns
nn_conservation_1D_stats = {} #(point_count, mapping, radius) -> (avg, var)
#knn_conservation_cores #[][][][] # points x mappings x knncs x cc
#knn_conservation_cores_stats #[][][][] # points x mappings x knncs x cc
#nn_conservation_cores #[][][][] # points x mappings x nns x cc
#nn_conservation_cores_stats #[][][][] # points x mappings x nns x cc


# Run experiment batch.

def main():
	if verbosity > 0:
		print_parameters()

	# sort point counts.
	point_counts.sort()
	# sort nearness ranges and knn counts
	k_values.sort()
	nearness_radii.sort()
	max_pc = point_counts[-1]
	max_knnc = k_values[-1]
	# initialize distance matrix
	distance_matrix = [[0 for j in xrange(max_pc)] for i in xrange(max_pc)]

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
		if verbosity > 1:
			print "Coordinates:"
			for c in enumerate(coordinates):
				print c
			print
		if verbosity > 3:
			print 'Distance matrix ({0}x{0}):'.format(point_count)
			printf_array(2, [""], distance_matrix) # TEST
			print

		for point in range(point_count):
			dists_from = distance_matrix[point] #selects row of matrix
			# Get lists of near neighbors
			for radius in nearness_radii:
				near_neighbors[(point_count, point, radius)] = [p for p in range(point_count) if dists_from[p] <= radius and p != i]
		if verbosity > 2:
			print "Near neighbors (point count, point, radius):\n",
			printf_dict(near_neighbors)
		# Get ordered list of nearest neighbors
			
		# metric: get avg. distance to nearest neighbors,
		# END

		# LOOP: for each mapping
		for map in mappings:
			try:
				if map == 'hilbert':
					# For each point, get 1D image
					images_1D = [(hilbert.hilbert_index(dimension, order, c[1]), c[0]) for c in enumerate(discretized_coords)]
					# Sort into 1D order by hilbert distance (called hilbert_index above). 
					# Each element is of form (hilbert_distance, coordinate_index) 
					images_1D.sort()
					if verbosity > 2:
						print "Hilbert images (Hilbert distance, particle #):\n", images_1D, "\n"
					ordering_1D = [images_1D[i][1] for i in range(len(images_1D))]
					if verbosity > 1:
						print "Hilbert ordering (by particle #):\n", ordering_1D, "\n"
				else:
					raise Exception("Map '{0}' not supported.\n".format(map))

		#     LOOP: for each nearest_neighbor counts
		#       metric: % of k nns within k steps in 1D ordering
		#     END
				
				# Metric: % of near neighbors (nns) within <# nns> steps of given point in 1D ordering
				for radius in nearness_radii:
					sum = 0.0	
					points_with_neighbors = 0.0
					for point in range(point_count):
						index_p = ordering_1D.index(point)
						#get near neighbors
						nns = near_neighbors[(point_count, point, radius)]
						nn_count = len(nns)
						if nn_count == 0:
							nn_conservation_1D[(point_count, point, map, radius)] = None
							nn_conservation_1D_stats[(point_count, map, radius)] = None
						else:
							#see how many near neighbors are within nns_count distance in 1D
							nn_count_1D = 0.0
							for neighbor in nns:
								index_n = ordering_1D.index(neighbor)
								dist = math.fabs(index_p - index_n)

								#is within len(nns) to point?
								if dist < nn_count:
									nn_count_1D += 1
							percentage = nn_count_1D/nn_count
							nn_conservation_1D[(point_count, point, map, radius)] = percentage
							sum += percentage
							points_with_neighbors += 1
					# Metric: Average and variance of above
					average = sum/points_with_neighbors
					variance_denom = 0.0
					for point in range(point_count):
						percentage = nn_conservation_1D[(point_count, point, map, radius)]				
						if percentage != None:
							difference = average - percentage
							variance_denom += (difference*difference)
					variance = variance_denom/points_with_neighbors
					nn_conservation_1D_stats[(point_count, map, radius)] = (average, variance)
				if verbosity > 1:			
					print "1D conservation of near neighbors:\n(point count, point, map, radius) : percentage by point\n",
					printf_dict(nn_conservation_1D)
				if verbosity > 0:
					print "Average/variance of 1D conservation of near neighbors:\n(point count, map, radius) : (average, variance)\n",
					printf_dict(nn_conservation_1D_stats)

				# LOOP: for each core count
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
			except Exception, e:
				print e



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

def printf_dict(d):
	for k, v in d.iteritems():
		print k, ": ", v
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
************************************\n""".format(hypercube_bound, dimension, mappings, order, point_counts, core_counts, k_values, nearness_radii, verbosity)

if __name__ == "__main__":
  main()
