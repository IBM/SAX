# XRF: eXplainable Random Forest

## Abstract
> Advanced machine learning models have become widely adopted in various domains due to their
> exceptional performance. However, their complexity often renders them difficult to interpret,
> which can be a significant limitation in high-stakes decision-making scenarios where
> explainability is crucial. In this study we propose eXplainable Random Forest (XRF), an
> extension of the Random Forest model that takes into consideration, crucially, during
> training, explainability constraints stemming from the users’ view of the problem and its
> feature space. While numerous methods have been suggested for explaining machine learning
> models, these methods often are suitable for use only after the model has been trained.
> Furthermore, the explanations provided by these methods may include features that are not
> human-understandable, which in turn may hinder the user’s comprehension of the model’s
> reasoning. Our proposed method addresses these two limitations. We apply our proposed method
> to six public benchmark datasets in a systematic way and demonstrate that XRF models manage
> to balance the trade-off between the models’ performance and the users’ explainability
> constraints.

​
## Data Example:
|amount             |credit_score       |risk               |is_credit|is_skilled|done_accept|
|-------------------|-------------------|-------------------|---------|----------|-----------|
|2252.039931401223  |3.222857167831561  |                   |True     |False     |True       |
|1305.5568953583925 |-3.2806098957653527|                   |True     |True      |False      |
|1067.548227650406  |0.3607771854432849 |                   |True     |True      |False      |
|-304.44489235514357|                   |0.672529318111246  |False    |True      |False      |
|805.8282582399206  |                   |0.7153447736537529 |False    |True      |False      |
|751.6687145117093  |                   |0.8049364265338377 |False    |True      |False      |
|942.0809629622781  |                   |0.7720897189668916 |False    |True      |False      |
|561.0995262233068  |                   |0.5834656266897115 |False    |False     |True       |
|969.3272817168503  |                   |0.6582375621427342 |False    |True      |False      |
​
​
## Software Requirements
> python3.7
>> numpy==1.19.2 \
>> scikit-learn==1.0.1 \
>> deap==1.3.1
​
​
## How to cite
Please consider citing [our paper](https://proceedings_paper.pdf) if you use code or ideas from this project:\
Amit, G., & Gur, S. (2024). eXplainable Random Forest. Human-Aware AI in Industry 5.0 at ECAI, ? Workshop Proceedings, ?, ?–?.
