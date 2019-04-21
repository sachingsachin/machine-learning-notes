# Machine Learning Notes

Machine learning enables computers to learn without being explicitly programmed.


# Supervised Learning

We present a data-set to the computer where each input/output pair is correct.
Note that the input above can be multi-dimensional and so can the output be too.
Supervised learning can be further broken down into two types:

1. **Regression Problem**: "Continuous-curve" fitting problem where the values corresponding to different inputs are very large in number. Example: number of candies sold per day for a whole year. Here the value "number sold per day" has a very wide range of values and so it would be considered a regression problem.

2. **Classification Problem**: In this case, the number of values corresponding to all the inputs are very few. Example: plotting of whether a person is retired or not against his age. In this case, there are only 2 outcomes: retired or not-retired. For such systems where the output values are easily countable, we identify them as classification problems. Note that the number of possible outcomes need not be limited to 2. Example, we can change the above to plot a person's status as student, working or retired against their age and it would still remain a classification problem because the number of output values is easily countable.


Put more formally, output in a regression problem varies continuosly while output in a classification problem is discrete-valued.


# Unsupervised Learning

In supervised learning, every point in our data-set is complete. It describes very precisely the value for inputs and outputs.
Where as in unsupervised learning, there is no output, only inputs. And its the job of the algorithm to find patterns and structures in that data to group them automatically. An example is to look through a catalog of thousands of images and try to group those images into categories such that similar looking images are grouped together. We only have input here, no proper definition of output is input to the algorithm. The algorithm itself tries to look for patterns for classification.


# Squared Error Function

For "continuous curve" problems, the simplest model is to draw a line such that it is as close to all the points as possible.
Mathematically, a line in 2D plane is expressed as:

h(x) = C<sub>0</sub> + C<sub>1</sub>x

So if the known set of values consists of data like [{x<sub>0</sub>, y<sub>0</sub>}, {x<sub>1</sub>, y<sub>1</sub>}...  {x<sub>i</sub>, y<sub>i</sub>}... {x<sub>m</sub>, y<sub>m</sub>}], then the **cost function** is defined as:

c = summation (h(x<sub>i</sub>) - y<sub>i</sub>)<sup>2</sup>/2M

And the line is considered a best fit when C<sub>0</sub> and C<sub>1</sub> are chosen to provide a minimum value for the above cost function.


# Contour plots for simplest error function
  
The squared error function `c` is a function of C<sub>0</sub> and C<sub>1</sub>.
Since there are three variables here, `c`, C<sub>0</sub> and C<sub>1</sub>, they are best shown in a 3D graph and then one can select the global minima in that 3D graph to find those C<sub>0</sub> and C<sub>1</sub> for which `c` is the minimum.

A 2D alternative to 3D graph is the [Contour line](https://en.wikipedia.org/wiki/Contour_line). Any contour line on a contour plot is the line joining those set of {C<sub>0</sub>, C<sub>1</sub>} points for which `c` has the same value.  
It can also be thought of as the projection of the 3D graph on a 2D surface.

Naturally, set of C<sub>0</sub> and C<sub>1</sub> for which `c` has large values will be quite high and so contour lines for those will form pretty big circles. And conversely, the set of C<sub>0</sub> and C<sub>1</sub> for which `c` has low values will be quite low and so contour lines for those will be smaller circles.

So it is easy to postulate that the best curve-fitting line is obtained by those C<sub>0</sub> and C<sub>1</sub> for which the contour line forms the smallest circle.
