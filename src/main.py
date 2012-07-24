# Author: Meghan Clark
# Date: 6/29/12
# Description: The main method for Cartographer.
# UGH I want to redesign so badly. This is slow and hideous - almost everything in main()?? Really?
# I'm going to reimplement with quadtrees and do away with, say, the distance matrix.
# Scaling is terrible right now - 2000 points is too slow, much less a million or billion.
# Bottleneck is checking how many near neighbors ended up neighbors in 1D.
# Can streamline. Many redundant passes are being done right now.
# It'd be even better in C, but maybe harder to debug. Java, at least, would be faster.
# However, those both require rewriting the hilbert curve/z-order code, writing quadtree 
# algorithms for knn and nn metrics.

import random
import math
#import pprint
from scurve import hilbert #hilbert_point, hilbert_index
from scurve import zorder #

# Get batch parameters
# These should come from a file at some point.
hypercube_edge_len = 1.0 #This shouldn't have an effect on outcome, but what if it does?
order = 7 #Does this affect the outcome? How should I determine this?
dimension = 3
distribution = 'uniform'
point_counts = [1000] #Point counts should be higher than core counts
core_counts = [10, 100]
mappings = ['hilbert', 'zorder']
k_values = [2, 3] #for k nearest neighbor metrics
nearness_percentages = [0.10, 0.20, 0.50] #percentage of edge_len for near neighbor metrics
verbosity = 1
measure_nn = True
measure_scattered_core = True

# DATA STRUCTURES
# Coordinates for each point. Points are indices (implicitly).
coordinates = []
# Dynamically drawn grid squares containing points. This is also how we'll 
# determine the curve order to use. Looks like it's quadtree time after all. 
# Sigh. Hack it for now.
discretized_coords = [] 
nearness_radii = []
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
#gauss parameters chosen such that 0<x<1 ~99% of the time
gauss_mu = hypercube_edge_len/2
gauss_sigma = (hypercube_edge_len/2)/3.0 # <x<1 ~99% of the time

MAX_DISPLAY = 10 #For things that take up lots of space (like matrix)
MAX_DISPLAY_C = 20 #For compact lists (neighbors, images, ordering, etc)

# Run experiment batch.

def main():
	if verbosity >= 1:
		print_parameters()

	# sort point counts.
	point_counts.sort()
	# sort nearness ranges and knn counts
	k_values.sort()
	nearness_percentages.sort()
	nearness_radii = [x*hypercube_edge_len for x in nearness_percentages]
	max_pc = point_counts[-1]
	max_knnc = k_values[-1]
	# initialize distance matrix
	distance_matrix = [[0 for j in xrange(max_pc)] for i in xrange(max_pc)]
        prev_point_count = 0

	# LOOP: for each point count
	for point_count in point_counts:
	# generate point_count points (or add to existing), and fill in distance matrix as you go along.
		#print "STARTING MATRIX\n"
		for i in range(point_count-prev_point_count):
			new_point = []
			d_new_point = []
			for j in range(dimension):
				if distribution == 'uniform':
					coord = random.uniform(0, hypercube_edge_len)
				elif distribution == 'gauss':
					coord = random.gauss(gauss_mu, gauss_sigma) % hypercube_edge_len
				else:
					print 'Distribution {0} unsupported. Using uniform.\n'.format(distribution)
					coord = random.uniform(0, hypercube_edge_len)
				new_point.append(coord)
				d_new_point.append(int(round(coord*100)))
			coordinates.append(tuple(new_point))
			# get the discretized coordinate. Will write algorithm later, hack for now.
			discretized_coords.append(tuple(d_new_point))
			# fill out distance matrix as you go
			i_adj = i + prev_point_count
			for j in range(i_adj):
				dist = euclidean_dist(coordinates[i_adj], coordinates[j])
				# take advantage of the symmetry
				distance_matrix[i_adj][j] = dist
				distance_matrix[j][i_adj] = dist
			distance_matrix[i_adj][i_adj] = 0
		if verbosity >= 2 and point_count <= MAX_DISPLAY_C:
			print "Coordinates:"
			for c in enumerate(coordinates):
				print c
			print
		if verbosity >= 4 and point_count <= MAX_DISPLAY:
			print 'Distance matrix ({0}x{0}):'.format(point_count)
			printf_array(2, [""], distance_matrix) # TEST
			print

		for point in range(point_count):
			dists_from = distance_matrix[point] #selects row of matrix
			# Get lists of near neighbors
			for radius in nearness_radii:
				percent_radius = radius/hypercube_edge_len
				near_neighbors[(point_count, point, percent_radius)] = [p for p in range(point_count) if dists_from[p] <= radius and p != point]
		if verbosity >= 3 and point_count <= MAX_DISPLAY_C:
			print "Near neighbors (point count, point, radius):\n",
			printf_dict(near_neighbors)
		# Get ordered list of nearest neighbors
			
		# metric: get avg. distance to nearest neighbors,
		# END

		#print "STARTING MAP\n"
		# LOOP: for each mapping
		for map in mappings:
			try:
				if map == 'hilbert':
					# For each point, get 1D image
					images_1D = [(hilbert.hilbert_index(dimension, order, c[1]), c[0]) for c in enumerate(discretized_coords)]
					# Sort into 1D order by hilbert distance (called hilbert_index above). 
					# Each element is of form (hilbert_distance, coordinate_index) 
					images_1D.sort()
					if verbosity >= 3 and point_count <= MAX_DISPLAY_C:
						print "Hilbert images (Hilbert distance, particle #):\n", images_1D, "\n"
					ordering_1D = [images_1D[i][1] for i in range(len(images_1D))]
					if verbosity >= 2 and point_count < 20:
						print "Hilbert ordering (by particle #):\n", ordering_1D, "\n"
				elif map == 'zorder':
					zmap = zorder.ZOrder(dimension, order) #must find a way to determine # of bits
					images_1D = [(zmap.index(list(c[1])), c[0]) for c in enumerate(discretized_coords)]
					images_1D.sort()
					if verbosity >= 3 and point_count <= MAX_DISPLAY_C:
						print "Z-curve images (Z-curve distance, particle #):\n", images_1D, "\n"
					ordering_1D = [images_1D[i][1] for i in range(len(images_1D))]
					if verbosity >= 2 and point_count < 20:
						print "Z-ordering (by particle #):\n", ordering_1D, "\n"
				else:
					raise Exception("Map '{0}' not supported.\n".format(map))

		#     LOOP: for each nearest_neighbor counts
		#       metric: % of k nns within k steps in 1D ordering
		#     END
				
				# Metric: % of near neighbors (nns) within <# nns> steps of given point in 1D ordering
				for radius in nearness_radii:
					#print "GETTING NN/POINT\n"
					p_sum = 0.0
					n_sum = 0.0
					points_with_neighbors = 0.0
					percent_radius = radius/hypercube_edge_len
					for point in range(point_count):
						index_p = ordering_1D.index(point)
						#get near neighbors
						nns = near_neighbors[(point_count, point, percent_radius)]
						nn_count = len(nns)
						if nn_count == 0:
							nn_conservation_1D[(point_count, point, map, percent_radius)] = (0, (None))
							nn_conservation_1D_stats[(point_count, map, percent_radius)] = (0, (None))
						else:
							#see how many near neighbors are within nns_count distance in 1D
							nn_count_1D = 0.0
							for neighbor in nns:
								index_n = ordering_1D.index(neighbor)
								dist = math.fabs(index_p - index_n)

								#is within len(nns) to point?
								if dist <= nn_count:
									nn_count_1D += 1
							percentage = nn_count_1D/nn_count
							nn_conservation_1D[(point_count, point, map, percent_radius)] = (nn_count, (percentage))
							p_sum += percentage
							n_sum += nn_count
							points_with_neighbors += 1
					# Metric: Average and variance of above
					#print "GETTING NN STATS\n"
					if points_with_neighbors != 0:
						p_average = p_sum/points_with_neighbors
						n_average = n_sum/points_with_neighbors
						variance_denom = 0.0
						for point in range(point_count):
							(nn_count, (percentage)) = nn_conservation_1D[(point_count, point, map, percent_radius)]				
							if percentage != None:
								difference = p_average - percentage
								variance_denom += (difference*difference)
						variance = variance_denom/points_with_neighbors
						nn_conservation_1D_stats[(point_count, map, percent_radius)] = (n_average, (p_average, variance))
					else:
						nn_conservation_1D_stats[(point_count, map, percent_radius)] = (0, (None, None))

				# LOOP: for each core count
				if measure_scattered_core:
					for core_count in core_counts:
						# Divide ordering into core_count chunks
						chunk_size = len(ordering_1D)/core_count
						pivots = get_pivots(chunk_size, point_count)
						for percent_radius in nearness_percentages:
							scattered_core_counts = []
							scc_denom = 0
							for p in range(point_count):
								used_cores = [0 for i in xrange(core_count)]
								nns = near_neighbors[(point_count, p, percent_radius)]
								#if len(nns) > 0:
								for neighbor in nns:
									index = ordering_1D.index(neighbor)
									assigned_core = index/chunk_size
								#	print index, assigned_core, "\n"
									used_cores[assigned_core] = used_cores[assigned_core] or 1
								scattered_core_counts.append(sum(used_cores))
								scc_denom += 1
							scc_average = sum(scattered_core_counts)/scc_denom
							print "Average num cores/neighborhood ({4}, {0}, {1}, {3}): {2}".format(map, percent_radius, scc_average, core_count, point_count)





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
		prev_point_count = point_count
	if verbosity >= 2 and point_count <= MAX_DISPLAY_C:			
		print "1D conservation of near neighbors:\n(point count, point, map, radius) : (# of neighbors, percentage by point)\n",
		printf_dict(nn_conservation_1D)
	if verbosity >= 1:
		print "Stats for 1D conservation of near neighbors:\n(point count, map, radius) : (Average # of neighbors, (average, variance))\n",
		printf_dict(nn_conservation_1D_stats)



# Dump data to file. Or print, whatever. Should probably have a verbosity flag.

# It is assumed that all points have same number of dimensions.
# Euclidean dist between n-dimensional points a and b: 
#   sqrt((a0-b0)^2 + (a1-b1)^2 + ... + (an-bn)^2)
def euclidean_dist(a, b):
  return math.sqrt(math.fsum([math.pow(a[i]-b[i], 2) for i in range(len(a))]))

def get_pivots(chunk, pc):
	indices = []
	count = 0 - chunk
	while count <= pc:
		count += chunk
		indices.append(count)

def printf_array(dimensions, labels, array):
	# Make sure dimensions matches # labels
	if dimensions == 2:
		for i in range(len(array)):
			for j in range(len(array[i])):
				print '{0:10.3f}'.format(array[i][j]),
			print

def printf_dict(d):
	keys = d.keys()
	keys.sort()
	for k in keys:
		print k, ": ", d[k]
	print

def print_parameters():
	print """Report verbosity: {8}\n
*************** PARAMETERS ***************
Hypercube boundary length: {0}
Hypercube dimension: {1}
Curves: {2}
Curve order: {3}
Point counts: {4}
Point distribution: {9}
Core counts: {5}
K-Nearest-Neighbor counts: {6}
Nearness neighborhoods: {7}
******************************************\n""".format(hypercube_edge_len, dimension, mappings, order, point_counts, core_counts, k_values, nearness_percentages, verbosity, distribution)

if __name__ == "__main__":
  main()
