r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1.FALSE = The training data and the Test data is disjoint thus, The machine 
was not trained on the test data, so the test data cannot help us calculate 
the in-sample error.The test data help us estimate the out-sample error.

2.FALSE = we can take a too small training group and get bad results because 
we will not have enough data to learn from, on the other hand we can take a 
too small test group that might not describe the data correctly and that will 
lead to bad test results even if the machine preforms well. 

3.TRUE = We use the test set only for evaluating the performance of the model
 not for choosing the best hyperparmeters of the model.

4.TRUE = In order to estimate the accuracy of our model after training on a fold we test it on our valid set so
 we can understand it's generalization error.
"""

part1_q2 = r"""
**Your answer:**
Our friend used the test set as the validation set thus, the lambda selected is over-fitting to the test data and does 
not help him estimate the generalization error. Moreover he used only one fold so his cross-validation is not good.
Our friend needs new data to test his final model, because he can't test the model on data he already used.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
As we can infer from the graph, the best k = 3, increasing k up to will 3 improved our accuracy,
from k=3 increasing k will cause a decrease in our accuracy.
for k=1 we will choose the sample that is the closest to us with no regard to other samples this will probably cause
under-fitting. On the other hand, for k equal to the number of elements we will get the most common label in our 
dataset. so choosing one of the extreme k values will probably harm us, but again its depend on the dataset.
"""

part2_q2 = r"""
**Your answer:**
1.Select the best model with respect to the training set accuracy will be like taking a test when you already know the
answers. So we will probably pass this exact test but a slightly different test will probably cause us to fail hard! 
Choosing our model based on the train set accuracy, will cause us to choose the model that best fit the data on it was 
trained. We will not be able to spot over-fitting because we will always get good performance for the train test 
accuracy. Since K-fold does not use the train-set for choosing the model, it allows evaluating the models with data that 
it was not trained with, that makes us capable of spotting over-fitting.

2. There is a possibility that one test set might be a bad representation of the data, causing us to wrongly evaluate 
the model. When we use K-fold, we use various test-sets to evaluate the model. every time we choose a different part of 
the dataset as the test-set that leads to a balanced solution, overcoming a test-set that not represent or data.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
We use Delta as a hyperparemeter, it is responsible for making sure that the model gives the true labels that are bigger
 in at least Delta from all other scores.
If we preserve that delta is positive, we can still get the same model if we will change the weights W and the 
regularization lambda to get the same model.

$\Delta$ is a hyperparameter and it's goal is to make sure that the score our model gave to the true label is greater 
by at least $\Delta$ from all other scores.
If we increase or decrease $\Delta$, as long as it is greater than 0, we can modify the weights W and the regularization 
$\lambda$ to express the same model.
"""

part3_q2 = r"""
**Your answer:**
1. Our linear model looks for some uniqe attributes per number between 0-9, for example:
our model classifies a number as 7 by looking for a sharp edge shape in the right corner of the picture. 
Our model made a mistake when interpreted a 2 as a seven due to it's sharp edge in the same place.
Our model also looks for a loop on the left of the number 2.
This is why the picture where the number 6 has a loop on his left was interpreted as a 2 by our model.

2. We can see a similarity in the goal of both models, they try to find the best label for each sample. 
but KNN is based on memorizing the data, when getting a new sample to evaluate it chooses the most frequent label out of 
k nearest neighbours. We can see that there is no real learning process since we simply remeber the data and compare it 
to what we already know, while in out model we have a learning phase that modify the weight when getting a data that has 
a better fit to the picture class, in this way we increase the weight of the important points (key-points) of the 
classes that allows better classification.
"""

part3_q3 = r"""
**Your answer:**
1. We think that our learning rate is good.
For too high learning rates we would see a sequence of minimum and maximum in the graph in opposite to the contiguous 
graph we got. High learning rate may cause to miss the minimum points and lead to lower accuracy and bigger loss.
For too low learning rates the changes of the weights in each epoch would be lower, 
as we can see in the graph the changes in each epoch are quite decent.

2. We can see that the model performed better on the training set (about 5 percent better), that means it is slightly 
over-fitting the data set. we would say it is slight over-fit since we still get good rates for the test set.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
We would like to see the points spread around the zero-line, which means that the prediction is correct (y = y hat).
We can observe that the final plot after CV is much better than the results from the top-5 features plot since the 
points clustered much closer around the zero line.

"""

part4_q2 = r"""
**Your answer:**
1. The training algorithm is still linear, the model is only linear with respect to the features/polynomials used.
If the question is with respect to x so of course the model is not linear because x^2 is not linear.

2. Yes, the model could be represented using a polynomials- something like a tailor series.
But as we know finding good representation like that isn't always easy, it may demand using high degree polynomials 
which will demand large computational power.

3. If we apply non linear features we make the decision boundary non-linear, because we apply a non-linear function on 
the input.With respect to the higher dimension we got after applying the function the boundary is linear but, 
with respect to the original input the boundary is nonlinear.
"""

part4_q3 = r"""
**Your answer:**
1. We dont know the best magnitude for lambda thus, using log function helps us with choose lambda with different magnitudes
By doing so we widen our hyper-parameter range and potentially achieve a better model with better correlations.

2. The model was fitted to the data for each lambda, for each degree and for each fold, so in total:
$lambda-range *num-of-degrees*k = 20*3*3 = 180$ 
"""

# ==============
