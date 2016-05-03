#!/bin/python

##
### Make_Geodesic_2.py
###
### Example 2 for Discrete Wasserstein Geodesics
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
	end_measure = []
	end_total = (2*31*5) + (21*5*2)
	end_weight = 1.0/end_total
	end_measure += BS.Make_Rectangle(0.1,0.1,31,5,end_weight)
	end_measure += BS.Make_Rectangle(0.1,0.36,31,5,end_weight)
	end_measure += BS.Make_Rectangle(0.1,0.15,5,21,end_weight)
	end_measure += BS.Make_Rectangle(0.36,0.15,5,21,end_weight)

	# Build starting measure
	start_measure = []
	start_total = (31*5) + (5*13) + (5*13)
	start_weight = 1.0/start_total
	start_measure += BS.Make_Rectangle(0.6,0.73,31,5,start_weight)
	start_measure += BS.Make_Rectangle(0.73,0.6,5,13,start_weight)
	start_measure += BS.Make_Rectangle(0.73,0.78,5,13,start_weight)
	
	# Make sure measures' masses add to 1.0 exactly  
	BS.Fix_Distribution_Mass(start_measure)
	BS.Fix_Distribution_Mass(end_measure)

	print "Measures created..."

	# Initialize geodesic object
	# LP is solved in this step
	geodesic = BS.Wasserstein_Geodesic(start_measure, end_measure)

	
	# Create geodesic image
	print "Printing Geodesic..."
	geodesic.Plot_Geodesic(3,"geodesic_figure_check_2")
	
	# Print time elapsed
	t = time.clock() - t
	print "Total Time elapsed: ", t

	return 1

## main()
#

# Runs script
if __name__ == '__main__':
	main()
