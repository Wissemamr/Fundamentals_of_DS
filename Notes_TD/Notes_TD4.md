### Exercise 1 : 
- What do we mean by stratified sampling? What does strata set means?

Stratified sampling is a sampling method that involves dividing a population into smaller subgroups called strata. These strata are formed based on shared attributes or characteristics of the population, such as age, gender, income level, or educational attainment. The purpose of stratified sampling is to ensure that the sample obtained is representative of the entire population being studied.
By dividing the population into strata, researchers can ensure that each subgroup is adequately represented in the sample, even if certain subgroups are relatively small in size.
Once the population is divided into strata, a probability sampling method, such as simple random sampling or systematic sampling, is used to select samples from each stratum. This ensures that each individual within the population has an equal chance of being selected for the sample.
Stratified sampling is particularly useful when there are significant variations or differences within the population.
<b> Strata Set </b>
In the context of stratified sampling, a strata set refers to the collection of subgroups or strata into which the population is divided. Each stratum within the strata set represents a specific attribute or characteristic of the population.

For example, if researchers are conducting a survey on income levels and educational attainment, they may divide the population into strata based on different levels of education (e.g., high school, college, postgraduate) and further divide each education level into income brackets (e.g., low income, middle income, high income). In this case, the strata set would consist of the different combinations of education levels and income brackets.


- What problem bootstrapping solves? , and which limitation does it suffer from?


Bootstrapping is a technique that involves resampling a dataset to estimate the uncertainty of a statistical parameter or to create multiple training sets for model training.
Estimating Parameter Uncertainty: Bootstrapping allows us to estimate the sampling distribution of a statistic or parameter when the underlying distribution is unknown or difficult to model. By repeatedly sampling from the original dataset with replacement, we can generate multiple bootstrap samples and calculate the statistic of interest for each sample. This provides an empirical approximation of the sampling distribution, which can be used to estimate confidence intervals or perform hypothesis testing.

Model Training and Evaluation: Bootstrapping can be used to create multiple training sets by resampling the original dataset. This is particularly useful when the dataset is small or imbalanced, as it allows us to generate diverse training sets and reduce the risk of overfitting. By training models on different bootstrap samples and aggregating their predictions, we can obtain more robust and reliable estimates of model performance.

Dealing with Limited Data: Bootstrapping can help mitigate the limitations of small or limited datasets by generating additional synthetic data. By resampling the available data with replacement, we can create new samples that are similar to the original data distribution. This can be particularly useful in scenarios where collecting more data is expensive or time-consuming.


- In model selection, which criterion we care for to select the best
model?

Model performance and model complexity
Probabilistic model selection (or “information criteria”) provides an analytical technique for scoring and choosing among candidate models.

Models are scored both on their performance on the training dataset and based on the complexity of the model.

Model Performance. How well a candidate model has performed on the training dataset.
Model Complexity. How complicated the trained candidate model is after training.