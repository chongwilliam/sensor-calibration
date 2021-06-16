import os 
import numpy as np
import matplotlib.pyplot as plt 

def main(filename, force_direction, test_number):
	""" Generates principal axes and magnitudes """ 
	data = np.genfromtxt(filename, delimiter=",")  # (t, fxL, fyL, fzL, mxL, myL, mzL, fxR, fyR, fzR, mxR, myR, mzR)
	save_filename = "./processed_data/may_21_2021_pool/" + test_number + "_calibration.csv"

	# Left force sensor
	left_data = data[:, 1:4]
	left_info = get_principal_axes(left_data)  # left 
	plot_data(left_data, left_info, force_direction, save_filename[:-4] + "_left.png")	

	# Right force sensor
	right_data = data[:, 7:10]
	right_info = get_principal_axes(right_data)  # left 
	plot_data(right_data, right_info, force_direction, save_filename[:-4] + "_right.png")

	# Save calibration data (overwrite if exists)
	if os.path.isfile(save_filename):
		os.remove(save_filename)

	with open(save_filename, "w") as f:
		np.savetxt(f, np.array(["Left Force Sensor"]), delimiter=" ", fmt="%s")	
		np.savetxt(f, left_info[0], delimiter=",", header="Axis 1", fmt="%4f")
		np.savetxt(f, left_info[1], delimiter=",", header="Axis 2", fmt="%4f")
		np.savetxt(f, left_info[2], delimiter=",", header="Axis 3", fmt="%4f", footer="---------------")
		np.savetxt(f, np.array(["Right Force Sensor"]), delimiter=" ", fmt="%s")		
		np.savetxt(f, right_info[0], delimiter=",", header="Axis 1", fmt="%4f")
		np.savetxt(f, right_info[1], delimiter=",", header="Axis 2", fmt="%4f")
		np.savetxt(f, right_info[2], delimiter=",", header="Axis 3", fmt="%4f")


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
	return [u[:,0], u[:,1], u[:,2], *s]

def plot_data(data, info, force_direction, filename):
	""" Plots data with principal axes 
	Args:
		data: (N, 3)
		info: [ex, ey, ez, mx, my, mz] 	
	"""	
	# Get max length to scale quiver
	centroid = np.mean(data, axis=0)	
	data_centered = data - centroid
	max_length = np.max(data_centered)

	# Setup plot
	ax = plt.figure().add_subplot(projection="3d")
	ax.scatter(data[:,0], data[:,1], data[:,2], marker="*")  # plot point cloud	
	# ax.scatter(data_centered[:,0], data_centered[:,1], data_centered[:,2], marker="*")  # plot point cloud (centered)

	# Plot principal axes 
	ax.quiver(centroid[0], centroid[1], centroid[2], info[0][0], info[0][1], info[0][2], color="red", length=2*max_length)  # first principal
	ax.quiver(centroid[0], centroid[1], centroid[2], info[1][0], info[1][1], info[1][2], color="blue", length=2*max_length)  # second principal
	ax.quiver(centroid[0], centroid[1], centroid[2], info[2][0], info[2][1], info[2][2], color="green", length=2*max_length)  # third principal 
	# ax.quiver(0, 0, 0, info[0][0], info[0][1], info[0][2], color="red", length=2*max_length)  # first principal
	# ax.quiver(0, 0, 0, info[1][0], info[1][1], info[1][2], color="blue", length=2*max_length)  # second principal
	# ax.quiver(0, 0, 0, info[2][0], info[2][1], info[2][2], color="green", length=2*max_length)  # third principal 

	# Plot applied force direction
	ax.quiver(centroid[0], centroid[1], centroid[2], force_direction[0], force_direction[1], force_direction[2], \
				color="black", length=2*max_length)  # applied force direction 
	# ax.quiver(0, 0, 0, force_direction[0], force_direction[1], force_direction[2], color="black", length=2*max_length)  # applied force direction (centered)

	# Style	
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
	# plt.show()  # debug only

	# Save plot
	plt.savefig(filename)

if __name__ == "__main__":
	filename = "./raw_data/may_21_2021_pool/record_force_sensors_0.400000recordHz_3.csv"
	test_number = "3"
	force_direction = np.array([0, 0, 0])
	main(filename, force_direction, test_number)