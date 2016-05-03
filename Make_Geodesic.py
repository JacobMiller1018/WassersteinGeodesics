#!/bin/python

##
### Make_Geodesic_2.py
###
### Example 1 for Discrete Wasserstein Geodesics
###
### Author: Jacob A. Miller
### Last Edited: April 10, 2016
##

import Barycenter_Scripts as BS
import time

##
## Main script function
##
def main():

	# Clock total time elapsed
	t = time.clock()

	# Build ending measure
	end_measure = [[1/(5.0*81.0),[0.01*i + 0.5, 0.01*j + 0.5]] for i in range(-40,41) for j in range(-2,3)]

	# Build starting measure
	start_measure = [[1/(5.0*81.0),[0.01*j + 0.5, 0.01*i + 0.5]] for i in range(-40,41) for j in range(-2,3)]

	# Make sure measures' masses add to 1.0 exactly  
	BS.Fix_Distribution_Mass(start_measure)
	BS.Fix_Distribution_Mass(end_measure)

	print "Measures created..."

	# Initialize geodesic object
	# LP is solved in this step
	geodesic = BS.Wasserstein_Geodesic(start_measure, end_measure)

	# Create geodesic image
	print "Printing Geodesic..."
	geodesic.Plot_Geodesic(3,"geodesic_figure_check_1")
	
	# Print time elapsed
	t = time.clock() - t
	print "Total Time elapsed: ", t

	return 1

## main()
#

# Runs script
if __name__ == '__main__':
	main()

