#!/bin/python
##
### Barycenter_Scripts.py
###
### Scripts for calculating discrete 2-Wasserstein distance and discrete 2-Wasserstein Geodesics 
###
### Author: Jacob A. Miller
### Last Edited: May 2, 2016
##

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pulp
import math

##
## Calculuates the square Euclidean distance between two points specified as lists of coordinates
##
def Square_L2(point_1,point_2):

	if len(point_1) != len(point_2): # Check if points are in the same dimension
		raise TypeError("point_1 and point_2 are not the same dimension")
		return 0
	else:
		return sum([(point_1[i] - point_2[i])**2 for i in range(0,len(point_1)) ] ) 

## Square_L2()
#

##
## Fixes floating point error on distributions so that their mass sums to exactly one
## Distributions should be a list of points. A point P should have its mass as P[0] and its corrdinate location as P[1]. Its corrdinates should be given as a list/tuple.
## WARNING: This function will adjust the mass of the first point in the distribution.
## 
def Fix_Distribution_Mass(theDistribution):

	for i in range(0,3): # Repeat calculation multiple times to stabalize rounding errors 

		# Calculate total mass in the distribution
		total = 0.0
		for point in theDistribution:
			total += point[0]

		# Calculate error between mass and 1.0
		error = 1.0 - total

		if abs(error)>0.01:
			print "WARNING: Error on distribution's total mass is very high."

		# Adjust mass of the first point in the distribution
		theDistribution[0][0] += error

	return 1

## Fix_Distribution_Mass()
#


##
## Makes a Rectangular distribution size (x_steps x y_steps). The point (x_start,y_start) specifies the bottom left-hand coordinate's value, and x_step and y_step specifices the spacing between each point.
## Each point will have mass specified by weight
##
def Make_Rectangle(x_start,y_start,x_steps,y_steps,weight,x_step = 0.01,y_step = 0.01):

	return [[weight,[x_start + x_step*i, y_start + y_step*j]] for i in range(0,x_steps) for j in range(0,y_steps)]

## Make_Rectangle()
#

##
## Formulats the Wasserstein Transportation (the transportation problem used in earth mover's distance) between two different probability distributions
## Each distribution should be a list of points. A point P should have its mass as P[0] and its coordinate location as P[1]. Its coordinates should be given as list.
## Will return a PuLP problem which can be solved to obtain either the transport or the Wasserstein Distance
##
def Wasserstein_Transportation(Distribution_1, Distribution_2):

	left_margin = len(Distribution_1)
	right_margin = len(Distribution_2)

	print "Formulating LP: Problem size - m=", left_margin, " n=", right_margin
	
	# Check for empty distributions
	if left_margin == 0 or right_margin == 0:
		raise TypeError("Empty Distribution")
		return 0

	# Check if marginals sum to the same quantity (should be "1.0" for both)
	# Else return the difference
	left_marginal = [point[0] for point in Distribution_1]
	right_marginal = [point[0] for point in Distribution_2]
	if sum(left_marginal) != sum(right_marginal) or sum(left_marginal)<=0:
		print "left sum and right sum difference: ", sum(left_marginal) - sum(right_marginal)
		raise ValueError('Distribution marginals are unequal')
		return 0

	# Create a 2-dim array of Square-L2 Distance
	# This also checks that all points are in the same R^d by calling Square_L2 on all combinations 
	try:
		cost_matrix = [[Square_L2(point_1[1],point_2[1]) for point_2 in Distribution_2] for point_1 in Distribution_1]
	except Exception as e:
		print e
		raise ValueError('Invalid cost matrix')
		return 0

	# Create LP problem
	TransProblem = pulp.LpProblem('Wasserstein Distance Transportation LP',pulp.LpMinimize)
	
	print "Formulating LP: Making variable matrix..."

	#Create a 2-dim array of the LP variables 'y[i,j]'
	variable_matrix = []
	for i in range(0,left_margin):
		variable_vector = []
		for j in range(0,right_margin):
			variable_vector.append(pulp.LpVariable('y[' + str(i) + ',' + str(j) + ']',0))

		variable_matrix.append(variable_vector)
		

	Objective_coefficent_vector = [cost_matrix[i][j]*variable_matrix[i][j] for i in range(0,left_margin) for j in range(0,right_margin) ]

	print "Formulating LP: Adding objective function..."

	# Add objective cost function to the problem
	TransProblem += pulp.lpSum(Objective_coefficent_vector) 
	
	print "Forumlating LP: Adding constraints..."

	#Add constraint for each supplier in the left margin
	for i, x in enumerate(variable_matrix):
		TransProblem += pulp.lpSum(x) == left_marginal[i]

	#Add constraint for each demand point in the right margin
	for j, demand in enumerate(right_marginal):
		inFlow_vector = [variable_matrix[i][j] for i in range(0,left_margin)]
		TransProblem += pulp.lpSum(inFlow_vector) == demand

	print "Formulating LP: Returning problem..."

	return TransProblem

## Wasserstein_Transportation()
#

##
## Calculates the Wasserstein Distance between two distributions
## Each distribution should be a list of points. A point P should have its mass as P[0] and its coordinate location as P[1]. Its coordinates should be given as list.
## Returns 2-Wasserstein between the input distributions
## REQUIRES: GLPK (GNU Linear Programming Kit) installed
##
def Wasserstein_Distance(Distribution_1,Distribution_2):

	left_margin = len(Distribution_1)
	right_margin = len(Distribution_2)

	# Formulate required LP problem
	try:
		TransProblem = Wasserstein_Transportation(Distribution_1,Distribution_2)
	except Exception as e:
		print e
		raise ValueError("Could not formulate LP")
		return 0

	# Solve the given LP problem
	try:
		pulp.GLPK().solve(TransProblem)
	except Exception as e:
		print e
		raise ValueError("Could not solve LP")
		return 0

	return pulp.value(TransProblem.objective)

## Wasserstein_Distance()
#

##
## Extracts the flow variables from a solved Wasserstein Distance problem
## Input is a solved mxn - transportation problem with the name of each variable from souce i to demand j being 'y[i,j]', m, and n
## Output is a numpy mxn array with whose (i,j) coordinate is y[i,j] 
##
def Read_Wasserstein_Flow(TransProblem,m,n):

	flow_matrix = np.zeros([m,n])

	for v in TransProblem.variables():

		try:
			indicies = map(int,v.name[2:-1].split(',')) # Should return a tuple (i,j) if the variable name is 'y[i,j]'
		except:
			raise TypeError('Invalid index names')
			return 0

		flow_matrix[indicies[0]][indicies[1]] = v.varValue
	
	return flow_matrix

## Read_Wasserstein_Flow()
#

##
## Condenses the coordinates in a given distribution, so that points with the same Euclidean coordinates (within 'error' squared distance) are listed only once
## Each distribution should be a list of points. A point P should have its mass as P[0] and its coordinate location as P[1]. Its coordinates should be given as list.
## Output distributions will have the same format, but with condensed coordinates
##
def Condense_Distribution(theDistribution,error = 0.00000001):
	
	index = 0
	while index<len(theDistribution):
		tempIndex = index + 1
		temp = theDistribution[index][1]
		while tempIndex<len(theDistribution):
			if Square_L2(theDistribution[tempIndex][1],temp)<error:
				theDistribution[index][0] += theDistribution[tempIndex][0]
				del theDistribution[tempIndex]
			else:
				tempIndex += 1

		index += 1

	return 1
## Condense_Distribution()
#


##
## Plot a two dimensional distribution on a fixed axis object (pyplot)
## Each point will be ploted with a radius proportional to its mass. weightScale is the size of points with mass 0.5. Color determines the distribution's color
## 
def Plot_Distribution_Ax(theDistribution, ax, weightScale, color, op = 0.4):
	
	# Check if the distribution is 2-dimensional (using the first point)
	if len(theDistribution[0][1]) != 2:
		raise TypeError('Not a 2D distribution')
		return 0

	for point in theDistribution:

		radius = weightScale * point[0] * 2 # makes weightScale the size of circles for 0.5 weighted points
		ax.scatter(point[1][0],point[1][1],s=radius,c=color, edgecolor=color, alpha = op)

	return 1
## Plot_Distribution()
#

##
## Class object for a Wasserstein Geodesic between two discrete probability distributions
##
## Initialized with two distributions, specified as lists of points (see above functions)
## Given a lambda, it will then output the given point on the geodesic and its weighted mean objective value
##
class Wasserstein_Geodesic:

	## initializer
	## 
	## Solves and creates the necessary LP and saves all the required variables
	## Distribution_1 is the ENDING distribution in the geodesic
	## Distribution_2 is the STARTING distribution in the geodesic
	##
	def __init__(self,Distribution_2,Distribution_1):
		
		# Make the distributions condensed to make the LP problem easier to solve if possible
		Condense_Distribution(Distribution_1)
		Condense_Distribution(Distribution_2)

		self.left_margin = len(Distribution_1)
		self.right_margin = len(Distribution_2)
		
		# Formulated the Wasserstein Trasportation problem the two distribution
		# Flow variables from this will be used for the points on the geodesic
		try:
			print "Formulating LP..."
			tempLP = Wasserstein_Transportation(Distribution_1,Distribution_2)
		except Exception as e:
			print e
			raise ValueError('Cannot formulate LP')
		
		# Solve the returned transportation problem
		try:
			print "Solving LP..."
			pulp.GLPK().solve(tempLP)
		except Exception as e:
			print e
			raise ValueError('Cannot solve LP')

		
		self.supply_coordinates = [np.array(point[1]) for point in Distribution_1] # Get the coordinates of the final distribution
		self.demand_coordinates = [np.array(point[1]) for point in Distribution_2] # Get the coordinates of the starting distribution

		self.wasserstein_distance = pulp.value(tempLP.objective)
		# Save Wasserstein distance so the weighted sum of distance from the geodesic points to starting/ending distributions can be returned 

		self.flow_matrix = Read_Wasserstein_Flow(tempLP, self.left_margin , self.right_margin) # Read flow variables to formulate geodesic coefficents

	## __init__()
	#
	
	## 
	## Returns the weighted mean of the specified point on the geodesic by the parameter lambda
	##
	def Weighted_Mean(self,lambda_weight):
		
		# Bounds check lambda
		if lambda_weight<0 or lambda_weight>1:
			raise ValueError('Lambda value out of bounds')
			return 0

		return lambda_weight * (1.0 - lambda_weight) * self.wasserstein_distance

	## Weighted_Mean()
	#

	##
	## Returns the distribution of the specified point on the geodesic by the parameter lambda
	##
	def Geodesic_Point(self,lambda_weight):

		# Bounds check lambda
		if lambda_weight<0 or lambda_weight>1:
			raise ValueError('Lambda value out of bounds')
			return 0

		temp_Distribution = []

		for i in range(0,self.left_margin):
			for j in range(0,self.right_margin):

				if self.flow_matrix[i][j] > 0: # We only care about points in the support of our distribution (with non-zero mass)

					temp_Distribution.append( [ self.flow_matrix[i][j] , lambda_weight*self.supply_coordinates[i] + (1.0 - lambda_weight)*self.demand_coordinates[j] ]) 
		
		Condense_Distribution(temp_Distribution) # Condense the distribution because it's very likely we will have repeat coordinates for extreme values of lambda

		return temp_Distribution

	## Geodesic_Point()
	#

	##
	## Plots the distribution steps through a nxn grid (i.e. n^2 steps), and saves it as a single image (named 'filename')
	## n = nSteps
	## weightScale is the mass of any 0.5 mass points
	## baseColor/stepColor are the colors of the original distributions/steps between them
	## xaxis/yaxis and xdiv/ydiv are the bounds on the x/y axes and the number of tick marks on each
	## DPI is the digital pixel images on the outputed figure
	##
	def Plot_Geodesic(self, nSteps, filename, weightScale=150, baseColor = 'b', stepColor = 'r', xaxis=(0,1), yaxis=(0,1), xdiv=5, ydiv=5, DPI=300):

		# Checks there is more than one step
		if nSteps<2:
			raise ValueError("Not enough nxn steps. Must have n>1")
			return 0

		# Calculate lambda steps
		total_steps = nSteps**2
		step_size = 1.0/(total_steps - 1)
		lambda_steps = [ i*step_size for i in range(0,total_steps)]

		# Calculate each points on the geodesic
		geodesic_points = map(self.Geodesic_Point, lambda_steps)
		
		# Set up plot
		matplotlib.rc('font', family='Arial')
		fig, ax = plt.subplots( nSteps, nSteps, sharex = 'col', sharey = 'row')
		fig.set_size_inches(12,12)

		# Scale axes w/ tick marks
		x_step = float(xaxis[1] - xaxis[0])/xdiv
		y_step = float(yaxis[1] - yaxis[0])/ydiv
		x_steps = [i*x_step + xaxis[0] for i in range(1,xdiv)]
		y_steps = [i*y_step + yaxis[0] for i in range(1,ydiv)]
		plt.setp(ax, xticks = x_steps, yticks = y_steps) 

		# Recover starting and ending distributions
		start_distribution = geodesic_points[0]
		end_distribution = geodesic_points[-1]

		for i in range(0,nSteps):
			for j in range(0,nSteps):
				step_num = nSteps*i + j # calculates which step we're at

				Plot_Distribution_Ax(start_distribution, ax[i,j], weightScale, baseColor) # Plots the starting distribution 
				Plot_Distribution_Ax(end_distribution, ax[i,j], weightScale, baseColor) # Plots the ending distribution
				Plot_Distribution_Ax(geodesic_points[step_num], ax[i,j], weightScale, stepColor, op = 0.7) # Plots the step of the distribution
				ax[i,j].set_xlabel(u'\u03BB' + "=" + ('%.3f' % lambda_steps[step_num]) ) # Sets label for axis with lambda value listed

				# Adjust axis look to remove Top/Right boundries and ticks
				ax[i,j].spines['top'].set_visible(False)
				ax[i,j].spines['right'].set_visible(False)
				ax[i,j].get_xaxis().tick_bottom()
				ax[i,j].get_yaxis().tick_left()

				# Set axis limits 
				ax[i,j].set_xlim(list(xaxis))
				ax[i,j].set_ylim(list(yaxis))

		fig.savefig(filename, dpi=DPI)

		return 1
	## Plot_Geodesic()
	#

## Class Wasserstein_Geodesic()
#

##
## Saves an image of a two dimensional distribution to .png file with specified name
## weightScale is the size of the 0.5 mass point
## color is the radius surrounding the point, showing its mass
## xaxis/yaxis and xdiv/ydiv specify the size of the x/y axes and number of tick marks on them
## 
def Plot_Distribution(theDistribution,fileName, weightScale = 150, color = 'r', xaxis=(0,1), yaxis=(0,1), xdiv = 5, ydiv = 5):

	# Check if the distribution is 2-dimensional
	if len(theDistribution[0][1]) != 2:
		raise TypeError('Not a 2D distribution')
		return 0

	# Set up figure
	fig, ax = plt.subplots(1)
	fig.set_size_inches(12,12)

	# Set up tick marks
	x_step = float(xaxis[1] - xaxis[0])/xdiv
	y_step = float(yaxis[1] - yaxis[0])/ydiv
	x_steps = [i*x_step+xaxis[0] for i in range(1,xdiv)]
	y_steps = [i*y_step+yaxis[0] for i in range(1,ydiv)]
	plt.setp(ax, xticks = x_steps, yticks = y_steps) 

	# Remove top/right boundries/tick marks
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	# Set axis limits
	ax.set_xlim(list(xaxis))
	ax.set_ylim(list(yaxis))

	# Plot the distribution
	for point in theDistribution:

		radius = weightScale * point[0] * 2 # makes weightScale the size of circles for 0.5 weighted points
		ax.scatter(point[1][0],point[1][1],s=radius,c=color,alpha=0.4) # Plot surrounding point area showing mass
		ax.scatter(point[1][0],point[1][1],s=(7.0*math.sqrt(weightScale/150.0)),c='k',alpha=1.0) # Plot center support point, weird radius is LOTS of trial/error

	fig.savefig(fileName + '.png', dpi = 200)

	return 1

## Plot_Distribution()
#
