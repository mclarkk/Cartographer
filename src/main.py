# Author: Meghan Clark
# Date: 6/29/12
# Description: The main method for Cartographer.
# UGH I want to redesign so badly. This is slow and hideous - almost everything in main()?? Really?
# Scaling is terrible right now - 2000 points is too slow, much less a million or billion.
# Bottleneck is checking how many neighbors (near or k) ended up neighbors in 1D.
# Can streamline. Many redundant passes are being done right now.
# It'd be even better in C, but maybe harder to debug. Java, at least, would be faster.
# However, those both require rewriting the hilbert curve/z-order library code, or
# finding existing C libraries.

import random
import math
from scurve import hilbert
from scurve import zorder


# PARAMETERS

# Get batch parameters
# These should come from a file at some point.
hypercube_edge_len = 1.0 #This shouldn't have an effect on outcome, but what if it does?
order = 7 #Curve order/bits used. Used math (magic) to determine.
mappings = ['hilbert', 'zorder']
dimension = 2
distribution = 'gauss'
point_counts = [1000] #Point counts should be higher than core counts
core_counts = [10] #For cores-per-neighborhood metrics.
k_values = [4] #For k nearest neighbor metrics. Even nums are more accurate.
nearness_percentages = [0.01, 0.05, 0.1] #Proportion of edge_len for near neighbor metrics
verbosity = 0
runs = 10
measure_nn = True
measure_nn_core = False
measure_knn = False
measure_knn_core = False


# GLOBAL DATA STRUCTURES

final_knn_averages = [] #conservation_stats indexed by run
final_nn_averages = []
final_nn_core_averages = []
final_knn_core_averages = []
#gauss parameters chosen such that 0<x<1 ~99% of the time
gauss_mu = hypercube_edge_len/2
gauss_sigma = (hypercube_edge_len/2)/3.0 # <x<1 ~99% of the time

MAX_DISPLAY = 10 #For things that take up lots of space (like matrix)
MAX_DISPLAY_C = 20 #For compact lists (neighbors, images, ordering, etc)


# RUN EXPERIMENT

def main():
	print_parameters()

	# sort point counts.
	point_counts.sort()
	# sort nearness ranges and knn counts
	k_values.sort()
	nearness_percentages.sort()
	nearness_radii = [x*hypercube_edge_len for x in nearness_percentages]
	max_pc = point_counts[-1]
	max_knnc = k_values[-1]

	# OUTER LOOP FOR MULTIPLE RUNS
	for run in range(runs):
		print "Run {0}".format(run+1)

		# RUN-SPECIFIC DATA STRUCTURES
		# Coordinates for each point. Points are indices (implicitly).
		coordinates = []
		# Dynamically drawn grid squares containin 
		discretized_coords = [] 
		k_nearest_neighbors = {} #(pc, point, k) -> [] of neighbors
		near_neighbors = {} #(pc, point, radius) -> [] of neighbors
		knn_conservation = {} #
		knn_conservation_stats = {} #(point count, mapping, k value)
		nn_conservation = {} #
		nn_conservation_stats = {} #(point_count, mapping, radius) -> (avg, var)
		knn_core_conservation = {} #
		knn_core_conservation_stats = {} #
		nn_core_conservation = {} #
		nn_core_conservation_stats = {} #
		# initialize distance matrix
		distance_matrix = [[0 for j in xrange(max_pc)] for i in xrange(max_pc)]
		prev_point_count = 0
	
		# BEGIN RUN
		for point_count in point_counts:
		# generate point_count points (or add to existing), and fill in distance matrix as you go along.
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
				# get the discretized coordinate.
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
				printf_array(2, [""], distance_matrix)
				print

			for point in range(point_count):
				# Select a point's row in matrix
				dists_from = distance_matrix[point]
				# Get lists of near neighbors
				for radius in nearness_radii:
					percent_radius = radius/hypercube_edge_len
					near_neighbors[(point_count, point, percent_radius)] = [p for p in range(point_count) if dists_from[p] <= radius and p != point]
				# Get lists of k-nearest neighbors
				dist_point_pairs = [(dists_from[i], i) for i in xrange(len(dists_from))]
				dist_point_pairs.sort()
				for k in k_values:
					k_nearest_neighbors[(point_count, point, k)] = [dist_point_pairs[x+1][1] for x in xrange(k)]
			if measure_nn and verbosity >= 3 and point_count <= MAX_DISPLAY_C:
				print "Near neighbors (point count, point, radius):\n",
				printf_dict(near_neighbors)
			if measure_knn and verbosity >= 3 and point_count <= MAX_DISPLAY_C:
				print "K nearest neighbors (point count, point, k):\n",
				printf_dict(k_nearest_neighbors)
				
			# Now for each curve, map points to one dimension by ordering them by curve index.
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
						zmap = zorder.ZOrder(dimension, order)
						images_1D = [(zmap.index(list(c[1])), c[0]) for c in enumerate(discretized_coords)]
						images_1D.sort()
						if verbosity >= 3 and point_count <= MAX_DISPLAY_C:
							print "Z-curve images (Z-curve distance, particle #):\n", images_1D, "\n"
						ordering_1D = [images_1D[i][1] for i in range(len(images_1D))]
						if verbosity >= 2 and point_count < 20:
							print "Z-ordering (by particle #):\n", ordering_1D, "\n"
					else:
						raise Exception("Map '{0}' not supported.\n".format(map))

					# Metric: % of near neighbors (nns) within <# nns> steps of given point in 1D ordering
					if measure_nn:
						for radius in nearness_radii:
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
									nn_conservation[(point_count, point, map, percent_radius)] = (0, (None))
									nn_conservation_stats[(point_count, map, percent_radius)] = (0, (None))
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
									nn_conservation[(point_count, point, map, percent_radius)] = (nn_count, (percentage))
									p_sum += percentage
									n_sum += nn_count
									points_with_neighbors += 1
							# Average and variance of above across all points
							if points_with_neighbors != 0:
								p_average = p_sum/points_with_neighbors
								n_average = n_sum/points_with_neighbors
								variance_numerator = 0.0
								for point in range(point_count):
									(nn_count, (percentage)) = nn_conservation[(point_count, point, map, percent_radius)]				
									if percentage != None:
										difference = p_average - percentage
										variance_numerator += (difference*difference)
								variance = variance_numerator/points_with_neighbors
								nn_conservation_stats[(point_count, map, percent_radius)] = (n_average, (p_average, variance))
							else:
								nn_conservation_stats[(point_count, map, percent_radius)] = (0, (None, None))
					
					# Metric: Average num. of cores per near neighbor neighborhood
					if measure_nn_core:
						for core_count in core_counts:
							# Divide ordering into core_count chunks
							chunk_size = int(math.ceil(point_count/float(core_count)))
							pivots = get_pivots(chunk_size, point_count)
							# Find out how many cores a neighborhood gets scattered across
							for percent_radius in nearness_percentages:
								scattered_core_counts = []
								scc_denom = 0
								for p in range(point_count):
									used_cores = [0 for i in xrange(core_count)]
									nns = near_neighbors[(point_count, p, percent_radius)]
									if len(nns) != 0:
										for neighbor in nns:
											index = ordering_1D.index(neighbor)
											assigned_core = index/chunk_size
											if assigned_core < 0 or assigned_core >= core_count: #TEST
												print index, assigned_core, core_count #TEST
											used_cores[assigned_core] = used_cores[assigned_core] or 1
										scattered_core_counts.append(sum(used_cores))
										scc_denom += 1
								if scc_denom != 0:
									scc_average = sum(scattered_core_counts)/float(scc_denom)
								else:
									scc_average = None
								nn_core_conservation_stats[(point_count, map, percent_radius, core_count)] = scc_average

					
					# Metric: % of k nearest neighbors (knns) of a given point that are also knns in 1D ordering
					if measure_knn:
						for k in k_values:
							window_size = k+1
							total_1D_sum = 0.0
							for point in range(point_count):
								# Get list of knns in 1D ordering
								p_index = ordering_1D.index(point)
								if p_index < window_size/2: #k
									lower = 0
									upper = window_size
								elif p_index > (point_count - window_size/2): #pc - 2k
									lower = point_count - window_size
									upper = point_count
								else:
									lower = p_index-(window_size/2)
									upper = p_index+(window_size/2) + 1
								knns_1D = ordering_1D[lower:upper]
								# Is each n-space nearest neighbor a 1D nearest neighbor?
								knns = k_nearest_neighbors[(point_count, point, k)]
								knn_count_1D = 0
								for neighbor in knns:
									if neighbor in knns_1D:
										knn_count_1D += 1
								knn_conservation[(point_count, point, map, k)] = knn_count_1D/float(k)
								total_1D_sum += knn_count_1D 
							# Average and variance of above across all points
							average = total_1D_sum/(k*point_count)
							variance_numerator = 0.0
							for point in range(point_count):
								percentage = knn_conservation[(point_count, point, map, k)]				
								difference = average - percentage
								variance_numerator += (difference*difference)
							variance = variance_numerator/point_count
							knn_conservation_stats[(point_count, map, k)] = (average, variance)
							
				except Exception, e:
					import traceback
					print traceback.format_exc()
			
			#Update point count so previously generated points are preserved within run
			prev_point_count = point_count
		
		#Update stats across runs
		final_knn_averages.append(knn_conservation_stats)
		final_knn_core_averages.append(knn_core_conservation_stats)
		final_nn_averages.append(nn_conservation_stats)
		final_nn_core_averages.append(nn_core_conservation_stats)
		
		#Reporting
		if measure_nn and verbosity >= 3 and point_count <= MAX_DISPLAY_C:			
			print "1D conservation of near neighbors:\n(point count, point, map, radius) : (# of neighbors, percentage by point)\n",
			printf_dict(nn_conservation)
		if measure_nn and verbosity >= 1:
			print "Stats for 1D conservation of near neighbors:\n(point count, map, radius) : (Average # of neighbors, (average, variance))\n",
			printf_dict(nn_conservation_stats)
		if measure_nn_core and verbosity >= 1:
			print "Stats for cores/neighborhood (near):\n(point count, map, radius, core count) : average\n",
			printf_dict(nn_core_conservation_stats)
		if measure_knn and verbosity >= 3 and point_count <= MAX_DISPLAY_C:
			print "1D conservation of k nearest neighbors:\n(point count, point, map, k) : percentage by point\n",
			printf_dict(knn_conservation)
		if measure_knn and verbosity >= 1:
			print "Stats for 1D conservation of k nearest neighbors:\n(point count, map, k) : (average, variance)\n",
			printf_dict(knn_conservation_stats)
		if measure_knn_core and verbosity >= 1:
			print "Stats for cores/neighborhood (k nearest):\n(point count, map, k, core count) : average\n",
			printf_dict(knn_core_conservation_stats)
	
	#Now that you have the performances for each run, average the runs together!
	if verbosity >= 1:
		print "FINAL REPORT:"
	print "\n",
	# Average knn conservation across runs
	if measure_knn:
		sums = {}
		averages = {}
		v_sums = {}
		final_stats = {} #params => average, variance
		for r in range(runs):
			knn_averages = final_knn_averages[r]
			for params, stats in knn_averages.iteritems():
				if r==0:
					sums[params] = 0
				sums[params] += stats[0] #the average
		for params, total in sums.iteritems():
			averages[params] = total/runs
		for r in range(runs):
			knn_averages = final_knn_averages[r]
			for params, stats in knn_averages.iteritems():
				if r==0:
					v_sums[params] = 0
				difference = stats[0] - averages[params]
				v_sums[params] += difference*difference
		for params, v_sum in v_sums.iteritems():
			avg = averages[params]
			var = v_sum/runs
			final_stats[params] = (avg, var)
		print "Averages/Variances of k-nearest neighbor conservation over {0} runs:".format(runs)
		printf_dict(final_stats)
	# Average nn conservation across runs
	if measure_nn:
		sums = {}
		averages = {}
		v_sums = {}
		final_stats = {} #params => average, variance
		valid_runs = {}
		for r in range(runs):
			nn_averages = final_nn_averages[r]
			for params, stats in nn_averages.iteritems():
				if r==0:
					sums[params] = 0
					valid_runs[params] = 0
				a = stats[1][0] #the average for those params that run
				if a == None:
					a = 0
				else:
					valid_runs[params] += 1
				sums[params] += a
		for params, total in sums.iteritems():
			if valid_runs[params] == 0:
				averages[params] = None
			else:
				averages[params] = total/valid_runs[params]
		for r in range(runs):
			nn_averages = final_nn_averages[r]
			for params, stats in nn_averages.iteritems():
				if r==0:
					v_sums[params] = 0
				a = stats[1][0]
				average_a = averages[params]
				if a != None and average_a != None:
					difference = a - average_a
					v_sums[params] += difference*difference
		for params, v_sum in v_sums.iteritems():
			avg = averages[params]
			if valid_runs[params] == 0:
				var = None
			else:
				var = v_sum/valid_runs[params]
			final_stats[params] = (avg, var)
		print "Averages/Variances of near neighbor conservation over {0} runs:".format(runs)
		printf_dict(final_stats)
	# Average cores/neighborhood (nn) across runs
	if measure_nn_core:
		sums = {}
		averages = {}
		v_sums = {}
		final_stats = {} #params => average, variance
		valid_runs = {}
		for r in range(runs):
			nn_core_averages = final_nn_core_averages[r]
			for params, average in nn_core_averages.iteritems():
				if r==0:
					sums[params] = 0
					valid_runs[params] = 0
				if average == None:
					average = 0
				else:
					valid_runs[params] += 1
				sums[params] += average
		for params, total in sums.iteritems():
			if valid_runs[params] == 0:
				averages[params] = None
			else:
				averages[params] = total/valid_runs[params]
		for r in range(runs):
			nn_core_averages = final_nn_core_averages[r]
			for params, a in nn_core_averages.iteritems():
				if r==0:
					v_sums[params] = 0
				average_a = averages[params]
				if a != None and average_a != None:
					difference = a - average_a
					v_sums[params] += difference*difference
		for params, v_sum in v_sums.iteritems():
			avg = averages[params]
			if valid_runs[params] == 0:
				var = None
			else:
				var = v_sum/valid_runs[params]
			final_stats[params] = (avg, var)
		print "Averages/Variances of cores/neighborhood (near) over {0} runs:".format(runs)
		printf_dict(final_stats)

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
