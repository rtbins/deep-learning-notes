# Structuring machine learning projects

## ML strategy

Suppose for a cat classifier we got 90% accuracy. It needs to be improved for great customer experience. We start exploring and gets lots of ideas to improve the performance of the model,

- collect more cat data
- a more diverse training set
- train longer with gradient descent
- try Adam optimizers
- try a bigger or smaller network
- try dropout
- try L2 regularization
- modify network architecture (activation functions, hidden units)

Similarly, there can be different ways to improve this model, which if not chosen properly can take significant development time. For example, if a team chooses to collect more cats data, spend six months and get very fewer performance improvements, it will be a waste of time. It won't add much value. 
One needs an approach to properly select methods from the available one. Though the approaches in machine learning are continuously evolving, these techniques can help in properly evaluating and selecting methods efficiently.

## Orthogonalization

To fine tune a model many methods and approaches can be tried. An efficient ML person is good in selecting proper parameters to tune. Selecting these parameters plays an important role, one needs to select orthogonal parameters (knobs). For example, while designing a car if a manufacturer doesn't ensure that brake control does only break and doesn't add certain values to steering, it won't be an efficient car. These controls should be independent and orthogonal to each other.

### Chain of assumptions in ML

- Fit training set well on the cost function
  - to get human-level performance also depends on use cases
  - knobs: network size, type of optimizer(Adam)
- Fit dev set well on the cost function
  - knobs: regularization, a bigger training set
- Fit test sets well on cost functions
  - knobs: bigger dev set
- Performs well in the real world
  - knobs: change dev set, change cost function

  `early stopping` is less orthogonal as it simultaneously effects how well training set is fitting and effects performance on dev set.

## Setting up your goal

### Single number evaluation metric

When improving the performance of the cat model, whichever method one use, using a single number evaluation metric can simplify the iterative process. Consider the table below and assume one takes precision and recall both as a metric. While iterating over different versions, a model has to be evaluated based on these two number. Which can be tricky as one has to decide if model B having a better precision but less recall is actually better than model A. This can be simplified if F1 score is taken as a metric. From the table below it can be observed that considering the F1 score, model A is better.

Classifier | Precision | Recall | F1 Score
-----------|-----------|--------|---------
A          |95%        |90%     |92.4%
B          |98%        |85%     |91.0%

Consider one is evaluating the performance of the cat app among different demographics. An error rate of which is listed below. It will simplify the selection of the algorithm if one is evaluating based on the average error rate.
Algorithm | US  | China | India | Other | Average
--------- |-----|-------|-------|-------|--------
A         | 3%  | 7%    | 5%    | 9%    | 6%
B         | 5%  | 6%    | 5%    | 10%   | 6.5%
C         | 2%  | 3%    | 4%    | 5%    | 3.5%

### Satisficing and optimizing metrics

Consider a cat classifier. For the app one wants a model with high accuracy and also they care about how quickly the app runs (less running time). A metric can be defined in such a way that it combines both the metrics, equation 1.
> $$cost = accuracy - 0.1 * runtime \tag{1}$$
> $$maximizeAccuracy \mid runtime <= 100ms \tag{2}$$

In eqaution 2, accuracy is defined as `optimizing metric` and runtime is defined as `satisfying metric`.

Classifier | Accuracy | Running time
-----------|----------|-------------
 A         | 90%      | 80ms
 B         | 92%      | 95ms
 C         | 95%      | 1,500ms

Generally, if one has N metrics for a problem, one metric is optimizing metric while (N-1) metrics are satisficing metrics. For example, consider wakeword/ trigger model (Alexa, google home etc). Here one would like to increase accuracy, while decreasing false positive cases. Metric for this problem can be phrased as,
>maximize accuracy | (number of false positive) < 2,  every 24 hour

### Train/dev/test distribution

Selecting a dev set is like setting a target for the team. It should represent the data well. For example, if dev set only represents cat from US and UK, the model after training may not perform well on cat images from China or India. Dev set should have better representation of complete dataset. Changing a dev set is like placing placing the target to a different place. Which means optimization and training needs to be repeated again for the new target.

### Size of dev and test sets

Traditionally, for a machine learning problem train/test split was defined to be 80:20 split. This was considering the small dataset which is typically like 10,000, In deep learning problems, datasets are big and typically is with more than million data points. Therefore, size of test set should be big enough to give high confidence in overall performance. In these datasets test set can be as low as 1% because dataset comprising 1% should be enough to give confidence in the performance of the model. Some apps doesn't require high confidence. In those apps it is fine not to have a test set. Although having a test set reassure unbiased estimate about how well a model is performing before it is shipped.

### When to change dev/test sets and metrics

For a cat classifier, assume we have two models A and B. From the table below, Model A is better but it classify and include porn images in the predictions. Though 3% error rate is a good metric wrt dev set, this model is completely unacceptable for users of the app.
Algorithm | Error rate | Remark
----------|------------|---------
 A        | 3%         | Misclassify porn images as cat
 B        | 5%         | Doesn't include porn images

A new error metric can be defined for the model as below.
> $$Error=\frac{1}{n_{dev}}\sum_{i=1}^{n_{dev}} w^{(i)}\mathcal{L}\{y_{pred} \neq y^{(i)}\}\\
w^{(i)} =
\begin{cases}
1 & \text{if} & x^{(i)} & \text{is not porn} \\
10 & \text{if} & x^{(i)} & \text{is  porn} \\
\end{cases}
$$

For a model do the orthogonalization as follow

- Define a metric to evaluate the classifier, therefore tune the knob 1 by placing the target at a good location.
- Take the problem of optimizing along this metric as a seperate problem (knob 2)
- Change the dev set if model is performing well on metric and dev set but not in real world (for example, images shared by an user may not be of high quality but one used for training is).

### Comparing human-level performance

For a classifier, lowest achievable error is known as *Bayes optimal error*. It represent a very best mapping function which cannot be surpassed. For a given machine learning problem, *human level performance* are not much far from bayes optimal error. Therefore, it is more convinient to compare and aim towards human level performance while training a model. Human level performance is a *proxy* for bayes optimal error,

If for a model performance is less than human level performance then there are tool sets which can bridge this gap. For example, get labelled data from humans, better analysis of bias/variance, get analysis from mannual error analysis, analysing why did a person got this right.

While training a model we get two type of errors, train and dev. If difference between train and dev error is more than train and human level performance, then one should focus more on reducing train-dev error gap. This error difference between test-dev set is called *variance*. The methods discussed in previous section can be used here. If difference between test and human level performance is more, then this should be the focus. This is called as *Avoidable bias*.

In below table, for model 1, focus should be to reduce variance and in model 2 it should be to reduce avoidable bias.
Source        | Accuracy model 1  | Accuracy model 2
--------------|-------------------|-----------------
Team of humans| 0.5%              | 0.5%
One human     | 0.1%              | .1%
Train error   | 0.2%              | 0.3%
Dev error     | 0.8%              | 0.4%

When a model surpasses human level performance, the further improvement methods are not clear but it is possible. For problems like online advertisement, product recommendations, logistics (shipment time prediction), loan approvals etc the models performing better than humans are available. These problems are not a natural perception problem and involves structural data. Computers are better in examining these large dimensional data than humans. For problems involving natural perception, only recently models has achieved/surpassed human level performance (speech recognition, medical diagnostics etc).

#### Guidelines to improve model performance

For a supervised learning problem there are two fundamental assumptions. First, one can fit the train set pretty well therefore decreasing the avoidable biases. Second, training set performance generalize well to dev/test set, decreasing the variance. Observer the difference between bias and variance before doing the optimizations.

For reducing the avoidable bias consider

- Train bigger model
- Train longer, use better optimizer algorithm, RMSprop, Adam
- Try other neural network architectures, do hyperparameter search
- Decrease Regularization

For reducing the variance consider

- Collecting more data
- Regularization
- L2, dropout, data augmentation
- Try other neural network architectures, do hyperparameter search

### Error analysis

After training a model carrying out error analysis is important. It can be done by hand in an excel sheet. This analysis helps in selecting and prioritizing approaches along which a model can be optimized further. Manually examining the errors helps. For example, a cat classifier mis-classifying dogs as cat. Is it worthwhile for a team to spend six months creating a dog classifier? Following approaches can help in answering this.

- Get ~100 mislabelled dev set examples.
- Count up the number of dogs.
- If it is 5%, then removing this error will improve model performance only by 0.5% (initial performance 90%).
- Evaluate if this improvemnt is significant and desirable.

Ideas for cat detector

- Fix pictures of dogs being recognized as cat.
- Fix great cats being mis classified.
- Improve performance on blurry images.

Create a spreadsheet considering above and mannually analyzing misclassified images from dev sets. These percentages represent *ceiling* of improvements along the ideas.


Column A   | Dog      |Great cats| Blurry  | Comments
-----------|----------|----------|---------|---------
 1         |          |*         |         | out zoo
 2         |*         |          |         | pitbull
 3         |          |          |*        | rainy day
 .         |.         |.         |.        |.
 .         |.         |.         |.        |.
 % of total|8%        |51%       |41%      |100%

### Cleaning up incorrectly labelled data

Deep learning algorithms are robust to random errors (wrong key pressed) but are more prone to systematic errors. While doing error analysis add a column for incorrectly labeled data.

Column A   | Dog      |Great cats| Incorrect label| Comments
-----------|----------|----------|----------------|---------
 1         |          |          |*               | labeller missed cat in background
 2         |          |          |*               | drawing of a cat, not a real cat
 3         |          |          |*               | instagram filter
 .         |.         |.         |.               |.
 .         |.         |.         |.               |.
 % of total|8%        |51%       |41%             |100%

- If correcting this improve the model performance significantly, then it should be done.
- Same process should be applied to the dev and test set to make sure they belong to the same distribution.
- Also consider examining examples which model got right along with wrong one. Sometime it may be mislabelled and can introduce bias.
- Train and dev/test set after this transformations may now come from a slightly different distribution. Train data is often larger to apply above manual analysis. Different approaches as discussed below need to be followed here.

### Build first system quickly then iterate

For a machine learning problem, one must be aware about the different challenges. For example, for speech recognition noisy background, accented speech etch are the challenges. How to pick and address challenges is an important question. Building first system and iterating quickly is a good approach. Set up the dev/test set and setup a metric. Use bias/variance analysis and error analysis to prioritize next steps.