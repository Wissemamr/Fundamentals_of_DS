## Exercise 01:
1.  k= 1 -> Seedless
2. k=3 -> Seeded
3. k=5 -> Seeded 
4. dist = 11 (Euclidean distance)



## Exercise 02 : The least squared regression

### Question :
- Investigate where $a$ and $b$ formulas in $Y = aX + b $ come from
The formula : 
$$ a = \frac{Cov(X, Y)}{Var(X)^2}$$
$$ b = \bar{Y}- a \bar{X} $$
where $a$ is the slope and $b$ is the y-intercept


### Answer :
#### Definition :
The least squares method is a common approach used in linear regression to find the best-fitting line through a set of data points. The goal is to minimize the sum of the squared differences between the observed values (actual data points) and the values predicted by the linear model. This sum of squared differences is known as the "sum of squared errors" or "residual sum of squares" (RSS).


Here, $y$ is the dependent variable, 

x is the independent variable, $a$ is the slope of the line, and $b$ is the y-intercept. <br>
The goal is to find the values of $a$ and $b$ that minimize the sum of squared differences between the observed values $y_i$ and the values predicted by the model $ax_i + b$

Mathematically, this is expressed as:
$$\text{Minimize} \sum_{i=1}^{n} (y_i - (ax_i + b))^2$$


Expanding and simplifying this expression gives the residual sum of squares (RSS):
$$\text{RSS} = \sum_{i=1}^{n} (y_i - ax_i - b)^2$$
The least squares method aims to find the values of $a$ and $b$ that minimize this sum
The partial derivatives of the RSS with respect to $a$ and $b$ are set to zero, and the resulting system of equations is solved to find the values of $a$ and $b$ that minimize the RSS.

$$ \frac{\partial \text{RSS}}{\partial m} = -2 \sum_{i=1}^{n} x_i(y_i - mx_i - b) $$
$$ \frac{\partial \text{RSS}}{\partial b} = -2 \sum_{i=1}^{n} (y_i - mx_i - b) $$
































Links and ressources : 
- https://www.investopedia.com/terms/l/least-squares-method.asp#:~:text=The%20least%20squares%20method%20is%20a%20statistical%20procedure%20to%20find,the%20behavior%20of%20dependent%20variables.