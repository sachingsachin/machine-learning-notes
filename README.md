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

J = summation (h(x<sub>i</sub>) - y<sub>i</sub>)<sup>2</sup>/2M

And the line is considered a best fit when C<sub>0</sub> and C<sub>1</sub> are chosen to provide a minimum value for the above cost function.


# Contour plots for simplest error function
  
The squared error function `J` is a function of C<sub>0</sub> and C<sub>1</sub>.
Since there are three variables here, `J`, C<sub>0</sub> and C<sub>1</sub>, they are best shown in a 3D graph and then one can select the global minima in that 3D graph to find those C<sub>0</sub> and C<sub>1</sub> for which `J` is the minimum.

A 2D alternative to 3D graph is the [Contour line](https://en.wikipedia.org/wiki/Contour_line). Any contour line on a contour plot is the line joining those set of {C<sub>0</sub>, C<sub>1</sub>} points for which `J` has the same value.  
It can also be thought of as the projection of the 3D graph on a 2D surface.

Naturally, set of C<sub>0</sub> and C<sub>1</sub> for which `J` has large values will be quite high and so contour lines for those will form pretty big circles. And conversely, the set of C<sub>0</sub> and C<sub>1</sub> for which `J` has low values will be quite low and so contour lines for those will be smaller circles.

So it is easy to postulate that the best curve-fitting line is obtained by those C<sub>0</sub> and C<sub>1</sub> for which the contour line forms the smallest circle.



# Nature of cost function `J`

As can be seen above, `J` is a function of C<sub>0</sub> and C<sub>1</sub>.  
Technically it is also a function of y<sub>i</sub> and M etc kind of variables but those will not change once a problem is specified.
So for a specific problem, `J` is a function of C<sub>0</sub> and C<sub>1</sub> and the goal of machine learning algorithm is to find C<sub>0</sub> and C<sub>1</sub> such that `J` is minimized.

To understand how to find such C<sub>0</sub> and C<sub>1</sub>, let us consider for a moment that C<sub>0</sub> is zero and does not change.
Thus, our cost function `J` is now just a function of C<sub>1</sub>. Graphically, this gives us a line passing through 0,0 and whose slope C<sub>1</sub> needs to be found such that its distance from all the given points {x<sub>i</sub>, y<sub>i</sub>} is minimum (i.e. the squared error function has the least value). Obviously, there will be just one such slope C<sub>1</sub> (proof not provided here but it is pretty intuitive). So if we plot the graph of J as a function of C<sub>1</sub>, it will be a U-shaped curve like this:

![J as a function of C1](docs/ml-1.png)

When J is a function of both C<sub>0</sub> and C<sub>1</sub>, the situation is similar, just that the same U-shaped curve exists in two dimensions and looks somewhat similar to:

![J as a function of C0 and C1](docs/ml-2.png)


# Gradient Descent Algorithm

One of the algorithm to find C<sub>0</sub> and C<sub>1</sub> is the gradient descent algorithm. Again to start simple, let us consider the case when C<sub>0</sub> is zero and does not change. This leaves us with just C<sub>1</sub> and a single U-shaped curve in 2D shown above.

The gradient descent algorithm says that we can start with any value of C<sub>1</sub> and can calculate the next value of C<sub>1</sub> by the following function:
C<sub>1</sub> = C<sub>1</sub> - (α Δ)

Where **α** is a constant called the **step-size** (or the **learning rate**) and determines how fast we converge to our solution.  
And **Δ** is the slope of J at C<sub>1</sub>

So basically, all we are doing here is subtracting the a fraction of J's slope from C<sub>1</sub> to arrive at the next value of C<sub>1</sub>
The step-size **α** cannot be too high otherwise the convergence will never happen and the gradient descent algorithm will just keep overshooting the minima its planning to achieve. If **α** is too low, then the convergence will happen too slowly, thus wasting computation cycles.

With both C<sub>0</sub> and C<sub>1</sub>, the same thing applies:  
C<sub>0</sub> = C<sub>0</sub> - (α Δ<sub>0</sub>)  
C<sub>1</sub> = C<sub>1</sub> - (α Δ<sub>1</sub>)  

The important thing to note here is that Δ<sub>0</sub> and Δ<sub>1</sub> are the partial derivates of the function J(C<sub>0</sub>, C<sub>1</sub>) and in every iteration, they should be calculated before doing the above two computations i.e. following should not be done:  
C<sub>0</sub> = C<sub>0</sub> - (α Δ<sub>0</sub>)  
Update J(C<sub>0</sub>, C<sub>1</sub>) with new value of C<sub>0</sub>  
C<sub>1</sub> = C<sub>1</sub> - (α Δ<sub>1</sub>)  

Because with the above intermediate updation of `J`, the convergence is somewhat unexpected and the algorithm does not fall under the gradient descent algorithm category anymore.

# Calculation of the slope w.r.t x<sub>0</sub>, x<sub>1</sub> .. x<sub>n</sub>

Skipping the derivation part, the formula for slope Δ<sub>j</sub> is:

Δ<sub>j</sub> = summation<sub>i=1:m</sub>((h<sub>i</sub> - y<sub>i</sub>)\*(x<sub>i,j</sub>))/m  
Substituting value of h<sub>i</sub> =  
Δ<sub>j</sub> = summation<sub>i=1:m</sub>((x<sub>i</sub><sup>T</sup>C - y<sub>i</sub>)\*(x<sub>i,j</sub>))/m  


# Matrix Inverse

Just as in the non-matrix world, most numbers have an inverse such that AA<sup>-1</sup> = 1, similarly in the matrix world, there exists matrix inverse M<sup>-1</sup> such that MM<sup>-1</sup> = I (i.e. the identity matrix).

**Note**: Just like 0 does not have an inverse, every matrix does not have an inverse. Example:
```
  --   --
 | 0   0 |
 | 0   0 |
  --   --
```


# Matrix Transpose

Transpose matrix of any matrix is simply the matrix formed by converting each row of the original matrix into a column of the transpose matrix.
So if original matrix A is:
```
  --   --
 | 1   2 |
 | 3   4 |
 | 5   6 |
  --   --
```

Then A<sup>T</sup> would be:
```
  --      --
 | 1   3  5 |
 | 2   4  6 |
  --      --
```

# Multiple Linear Regression
In most real world situations, the cases are more complex than the simple **C<sub>0</sub> + C<sub>1</sub>x** situation.
A very natural extension of the above case is when:

h<sub>c</sub>(x) = C<sub>0</sub> + C<sub>1</sub>x<sub>1</sub> + C<sub>2</sub>x<sub>2</sub> ... + C<sub>n</sub>x<sub>n</sub>

Here, we can assume that x<sub>0</sub> is always 1.

Thus, in matrix notation, this can be presented as:

```
        --  
      | C0  |
      | C1  |
C =   | C2  |
        ...
      | Cn  |
        --  

        --  
      | x0  |
      | x1  |
X =   | x2  |
        ...
      | xn  |
        --  
```
and this gives H = C<sup>T</sup>X

Note that x<sub>0</sub>, x<sub>1</sub> ... x<sub>n</sub> are also called the N+1 **features** of the multiple linear regression


# Gradient Descent for multiple linear regression

Recall from the above that the squared error function J is defined as:

J = summation (h(x<sub>i</sub>) - y<sub>i</sub>)<sup>2</sup>/2M

This does not change for the multiple linear regression case. Just becomes a little more verbose as:

J(C<sub>0</sub>, C<sub>1</sub> ... C<sub>n</sub>) = summation (h<sub>c</sub>(x<sub>i</sub>) - y<sub>i</sub>)<sup>2</sup>/2M


The method to converge on the most desirable value set of C<sub>0</sub>, C<sub>1</sub> ... C<sub>n</sub> also remains the same:

C0 = C0 - (α Δ0)  
C1 = C1 - (α Δ1)  
...  
Cn = Cn - (α Δ1)  


# Feature Scaling

If the features x<sub>0</sub>, x<sub>1</sub> ... x<sub>n</sub> of multiple linear regression (MLR) vary widely from each other in their range of values, then you can imagine that the contour-plots for them would be very skewed. Think about the contour plot for just one feature x<sub>1</sub> whose H depends only on C<sub>0</sub> and C<sub>1</sub>. If range of C<sub>0</sub> is from 0 to 10 while range of C<sub>1</sub> is from 0 to 10000, then the contour plot between the two will skew very widely in the direction of C<sub>1</sub>.

It would be difficult to apply gradient descent in such a case because the gradient descent will keep overshooting the global minima for low-range C<sub>0</sub>. To solve this, we can divide the C<sub>0</sub>, C<sub>1</sub> by their ranges to get them both in the range `[-1, 1]`. When both come in the same range, their contour plot will look like a perfect circle and hence much easier to converge.

This division of C<sub>0</sub>, C<sub>1</sub> ... C<sub>n</sub> with their respective ranges is called `feature scaling`.

In practice, we do not just divide them with their respective ranges since its possible that one feature varies from 10,000 to 10,0001 while other varies from 0 to 10. If we just divide them by their ranges, it would be no good. So we apply the following formula:

x<sub>i</sub> = (x<sub>i</sub> - avg(x)) / (x<sub>max</sub> - x<sub>min</sub>)


# Polynomial Regression

It is not always possible to have a linear equation to represent the curve that will minimize the cost function i.e. fit the most number of points.
It could be a polynomial expression too like:

h(x) = C<sub>0</sub> + C<sub>1</sub>x + C<sub>2</sub>x<sup>2</sup> + C<sub>3</sub>x<sup>3</sup> ... C<sub>n</sub>x<sup>n</sup>

It turns out that we can easily reduce the above to a linear regression problem by redefining the features.  
For example, we can redefine as:  
z<sub>0</sub> = x<sup>0</sup>  
z<sub>1</sub> = x<sup>1</sup>  
z<sub>2</sub> = x<sup>2</sup>  
...  
z<sub>n</sub> = x<sup>n</sup>  

Thus, our curve h(x) becomes:  
h(x) = C<sub>0</sub> + C<sub>1</sub>z + C<sub>2</sub>z<sub>2</sub> + C<sub>3</sub>z<sub>3</sub> ... C<sub>n</sub>z<sub>n</sub>

which is same as linear regression ! So we solve for this linear regression and then transform back the results to the original form.


# Normal Equation : Solving for C by a formula

The below function J(C<sub>1</sub>) is minimum when its slope dJ/dC<sub>1</sub> = 0.  
This fact becomes obvious when we realize that only when slope=0, J(C<sub>1</sub>) changes direction from decreasing to increasing. So the minimum has to be when dJ/dC<sub>1</sub> = 0

![J as a function of C1](docs/ml-1.png)

The same fact applies when J is a function of C<sub>0</sub>, C<sub>1</sub> ... C<sub>n</sub> except that dJ/dC<sub>1</sub> becomes a [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative). So `J` achieves its lowest value when we solve all the following equations:  

  dJ/dC<sub>0</sub> = 0  
  dJ/dC<sub>1</sub> = 0  
  dJ/dC<sub>2</sub> = 0  
  ...  
  dJ/dC<sub>n</sub> = 0  

(Note that dJ/dC<sub>i</sub> denotes the partial derivative of J with respect to C<sub>i</sub>)

There are n+1 equations above and n+1 variables to solve.  
Solution for this kind of setup is given by the formula: C<sub>ideal</sub> = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y

where:  

`X` is m-by-n+1 input matrix  
`Y` is m-by-1 output matrix  
`m` is the number of data points we have.  
`n` is the number of features present in the the data.

Example: If we have the selling-price of a cloth as a function of n=3 features: size, cost price and tax with m=4 data points as follows:

|C<sub>1</sub>=size|C<sub>2</sub>=cost-price|C<sub>3</sub>=tax|y=selling-price|
|----|----------|---|-------------|
| 10 | 100      | 10| 140         |
| 11 | 120      | 15| 160         |
| 10 | 120      | 11| 155         |
| 14 | 140      | 14| 200         |

then our input matrix X<sub>m,n+1</sub> (with m = 4, n = 3) would be:
```
  --           --
| 1  10  100  10 |
| 1  11  120  15 |
| 1  10  120  11 |
| 1  14  140  14 |
 --            --
```
Here first column denotes C<sub>0</sub> which is always 1.

And y<sub>m,1</sub> would be:
```
 -- --
| 140 |
| 160 |
| 155 |
| 200 |
 -- --
 ```
 
Compared to gradient descent, this method of direct computation is good because:
1. There is no iteration involved and we get the best value of C(n,1) in one shot.
2. No need to find out a right value of step-size **α**
3. No need to do feature scaling

However, its only disadvantage is that the matrix operations involved here are of the O(n<sup>3</sup>) which makes it very costly when n is more than 10,000. At that point, it is sometimes advisable to fall back to gradient descent (with order as O(kn<sup>2</sup>)) to speed up calculations.


**Quick dimension check for (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y**

X is an (m, n+1) matrix.  
So X<sup>T</sup> would be (n+1, m) matrix.  
X<sup>T</sup>X would therefore be (n+1, m)(m, n+1) = (n+1, n+1) matrix.  
(X<sup>T</sup>X)<sup>-1</sup> would also be same (n+1, n+1) matrix.  
(X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup> = (n+1, n+1)(n+1, m) = (n+1, m) matrix.  
And the final expression (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y = (n+1, m)(m, 1) = (n+1, 1) matrix.  
(n+1, 1) matrix thus obtained gives us the ideal values of C<sub>0</sub>, C<sub>1</sub> ... C<sub>n</sub> that we need to best-fit the linear equation for the given data.
