import Barycenter_Scripts as BS
import time

def main():
	t = time.clock()
	start_measure = [[1/(5.0*81.0),[0.01*i + 0.5, 0.01*j + 0.5]] for i in range(-40,41) for j in range(-2,3)]
	end_measure = [[1/(5.0*81.0),[0.01*j + 0.5, 0.01*i + 0.5]] for i in range(-40,41) for j in range(-2,3)]
	
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
	geodesic.Plot_Geodesic(3,"geodesic_figure")
	
	t = time.clock() - t
	print "Time eslapsed: ", t

if __name__ == '__main__':
	main()
