import Barycenter_Scripts as BS
import time

def Make_Rectangle(x_start,y_start,x_steps,y_steps,weight,x_step = 0.01,y_step = 0.01):
	return [[weight,[x_start + x_step*i, y_start + y_step*j]] for i in range(0,x_steps) for j in range(0,y_steps)]

def main():
	t = time.clock()

	start_measure = []
	start_total = (2*31*5) + (21*5*2)
	sw = 1.0/start_total
	start_measure += Make_Rectangle(0.1,0.1,31,5,sw)
	start_measure += Make_Rectangle(0.1,0.36,31,5,sw)
	start_measure += Make_Rectangle(0.1,0.15,5,21,sw)
	start_measure += Make_Rectangle(0.36,0.15,5,21,sw)

	end_measure = []
	end_total = (31*5) + (5*13) + (5*13)
	ew = 1.0/end_total
	end_measure += Make_Rectangle(0.6,0.73,31,5,ew)
	end_measure += Make_Rectangle(0.73,0.6,5,13,ew)
	end_measure += Make_Rectangle(0.73,0.78,5,13,ew)
	
	for i in range(0,10):
		total = 0.0
		for point in start_measure:
			total += point[0]
		error = 1.0 - total
		start_measure[0][0] += error

		total = 0.0
		for point in end_measure:
			total += point[0]
		error = 1.0 - total
		end_measure[0][0] += error

	print "Measures created..."

	geodesic = BS.Wasserstein_Geodesic(start_measure, end_measure)

	print "Printing Geodesic..."
	geodesic.Plot_Geodesic(3,"geodesic_figure_2")
	
	t = time.clock() - t
	print "Time eslapsed: ", t

if __name__ == '__main__':
	main()
