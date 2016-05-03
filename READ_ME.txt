READ_ME.txt

Author: Jacob A. Miller
Last Edited: May 2, 2016


Contents:
I. 	Description
II. 	Included Files
III.	Requirements
IV. 	Running Examples
V.	Making Additional Examples

I. Description:

The files included herein are those to produce the Wasserstein geodesic and distance Examples in Sections 1.4 and 3.4 of the author's thesis. Included in these files are some basic functions/methods/classes for extending these examples. 

II. Included Files:
	a) Barycenter_Scripts.py - Python module with the Wasserstein distance function and Wasserstein geodesic class
	b) geodesic_figure.png - Wasserstein geodesic Example 1 and Figure 3.2 in author's thesis
	c) geodesic_figure_2.png - Wasserstein geodesic Example 2 and Figure 3.1 in the author's thesis
	d) Make_Geodesic.py - Python script to compute <geodesic_figure.png>. It will currently name this file <geodesic_figure_check_1.png>.
	e) Make_Geodesic_2.py - Python script to compute <geodesic_figure_2.png>. It will currently name this file <geodesic_figure_check_2.png>.
	f) READ_ME.txt - Current file

III. Requirements:

All scripts and modules were written and intended for Python 2.7.11. In addition to the standard class of Python modules (including <matplotlib>), these scripts require python's linear programming modeling module <puLP> and GLPK (the GNU linear programming kit) to solve its given LPs. On OSX, using homebrew and pip, these can be installed by:
	brew tap homebrew/science
	brew install glpk
and
	pip install pulp

IV. Running Examples:

In a local directory containing <Barycenter_Scripts.py> and either <Make_Geodesic.py> or <Make_Geodesic_2.py>, run the corresponding Example by
	python Make_Geodesic.py
	python Make_Geodesic_2.py

V. Making Additional Examples:

Additional examples can be make by using the format in <Make_Geodesic_2.py>. <Barycenter_Scripts.py> contains a function Make_Rectangle() which will allow you piece together a measure with various rectangular grids.

The sum of the masses at all points in the distribution must add to 1.0 (i.e. the <weight> parameter on each rectangle, times the number of points in the rectangle, summed up over all rectangles, must be 1.0). This is crucial to the feasibility of the LP, and GLPK will not find a solution to any example where this is not satisfied EXACTLY (even floating-point error is not allowed). Calling the function Barycenter_Scripts.Fix_Distribution_Mass() on the distribution will usually fix any floating point error issues, but will slightly skew your distribution (although it should not be noticeable in your final plot). It is suggested you always call this function on your distribution before initializing the Barycenter_Scripts.Wasserstein_Geodesic class object.

After defining your two distributions, the Barycenter_Scripts.Wasserstein_Geodesic class will solve the coefficients necessary to easily parameterize all points of the geodesic. You may then either have it return individual points in the distribution using the Geodesic_Point() method or plot multiple steps using the Plot_Geodesic() method. Note in Make_Geodesic and Make_Geodesic_2, these time these scripts take is mostly due to plotting and rendering the given figures (not solving the LP during the initialization of the class object). These can made to run much faster/slower by changing the number of steps on the Wasserstein Geodesic being printed or by changing the dpi of the final figure. See Barycenter_Scripts.py for explanation of the possible parameters of the Plot_Geodesic() method. 
