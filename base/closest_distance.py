import sys
sys.path.insert(0, 'classification/')
from ssa_features import *
from operator import itemgetter
from mapping import *

target = 'ValCivil'

def create_points(data_all, data_x, data_y):
	assert len(data_all) == len(data_x)
	assert len(data_x) == len(data_y)

	points = dict()
	for i in range(len(data_all)):
		points[data_all[i]] = (data_x[i], data_y[i])

	return points	

def euclidean(coords):
	xx, yy = ref
	x, y = coords
	return ((x-xx)**2 + (y-yy)**2)**0.5

def sort_by_euclidean_distance(data_all, data_x, data_y):
	points = create_points(data_all, data_x, data_y)
	xx, yy = points[target]

	points.pop(target, None)
	distances = dict()

	for (key, (x,y)) in points.items():
		dist = ((x-xx)**2 + (y-yy)**2)**0.5
		distances[key] = dist

	return sorted(distances.items(), key = itemgetter(1))

def flatten_binary(points, nr_times):
	flattened = []
	i = 0
	for (key, val) in points:
		if i == nr_times:
			break
		if key not in inv_mapping:
			feat = inv_mapping[inv_binary_convertor[key]]
			if feat not in flattened:
				flattened.append(feat)
				i += 1
		else:
			feat = inv_mapping[key]
			if feat not in flattened:
				flattened.append(feat)
				i += 1		
	return flattened			

def get_best(data_all, data_x, data_y, nr_times):
	points = sort_by_euclidean_distance(data_all, data_x, data_y)
	# print points
	return flatten_binary(points, nr_times)

if __name__ == "__main__":
	flattened = get_best(civil_net, civil_net_x, civil_net_y, 5)
	print flattened


