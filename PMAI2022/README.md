# Model-informed LIME Extension for BusinessProcess Explainability
## Abstract
> Our focus in this work is on the adaptation of eXplainable AI
> techniques for the interpretation of business process execution results. 
> Such adaptation is required since conventional employment of suchtechniques
> involves a surrogate machine learning model that is trained on historical process
> executionlogs. However, being a data-driven surrogate, its representation
> faithfulness of the real business process model affects the adequacy of the 
> explanations derived from it.  Hence, native use of such techniquesis not 
> ensured to be adhering to the target business process explained. 
> We present a business-process-model-driven approach that extends LIME,
> a conventional machine-learning-model-agnostic eXplain-able AI tool,
> to cope with business processes constraints that is replicable and reproducible. 
> Our results show that our extended LIME approach produces correct and significantly 
> more adequate explanationsthan the ones given by LIME as-is.
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
> numpy==1.21.2 \
> scipy==1.7.3\
> scikit-learn==1.0.1 \
> pandas==1.3.3 \
> lime==0.2.0.1
​
​
## How to cite
Please consider citing our paper if you use code or ideas from this project: