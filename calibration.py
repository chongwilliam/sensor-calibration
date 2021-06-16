import numpy as np
import matplotlib.pyplot as plt 

def main(filename):
	""" Generates principal axes and magnitudes """ 
	data = np.genfromtxt(filename, delimiter=",")  # (t, fxL, fyL, fzL, mxL, myL, mzL, fxR, fyR, fzR, mxR, myR, mzR)

	# Left force sensor
	left_data = data[:, 1:4]
	info = get_principal_axes(left_data)  # left 
	print(info)
	# np.savetxt("calibration.csv", info, delimiter=",")
	plot_data(left_data, info)	

	# Right force sensor
	right_data = data[:, 7:10]
	info = get_principal_axes(right_data)  # left 
	# np.savetxt("calibration.csv", info, delimiter=",")
	plot_data(right_data, info)


def get_principal_axes(data):
	""" PCA on covariance matrix to obtain principal axes 
	Args:
		data: (N, 3) 
	Output:
		[ex, ey, ez, mx, my, mz]  (e: unit vector, m: magnitude)
	"""

	centroid = np.mean(data, axis=0)	
	data_centered = data - centroid
	u, s, vh = np.linalg.svd(data_centered.T)
	# data_cov = np.matmul(data_centered.T, data_centered)
	# evals, evecs = np.linalg.eig(data_cov)
	# sort from highest to lowest 
	# sort = s.argsort()[::-1]
	# s = s[sort]
	# evecs = evecs[:,sort]
	# return [*evecs, *evals]
	return [u[:,0], u[:,1], u[:,2], *s]

def plot_data(data, info):
	""" Plots data with principal axes 
	Args:
		data: (N, 3)
		info: [ex, ey, ez, mx, my, mz] 	
	"""	

	centroid = np.mean(data, axis=0)	
	data_centered = data - centroid
	# get max length to scale quiver
	max_length = np.max(data_centered)
	ax = plt.figure().add_subplot(projection="3d")
	ax.scatter(data[:,0], data[:,1], data[:,2], marker="*")  # plot point cloud	
	# plot principal axes 
	ax.quiver(centroid[0], centroid[1], centroid[2], info[0][0], info[0][1], info[0][2], color="red", length=2*max_length)  # first principal
	ax.quiver(centroid[0], centroid[1], centroid[2], info[1][0], info[1][1], info[1][2], color="blue", length=2*max_length)  # second principal
	ax.quiver(centroid[0], centroid[1], centroid[2], info[2][0], info[2][1], info[2][2], color="green", length=2*max_length)  # third principal 
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
	plt.show()

if __name__ == "__main__":
	filename = "./raw_data/may_21_2021_pool/record_force_sensors_0.400000recordHz_0.csv"
	main(filename)