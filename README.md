# RFE
Recursive Feature Elimination (RFE) algorithm to remove potentially redundant gene contributions that can overfit the model for the risk prediction. 


## Installation
You can use BigPipeline environment.
* conda activate pyBigPipeline


## Usage
Required arguments:
| Argument                | Descripion |
| ---------------------- |--------------------- |
| --input_train |Path to pEA matrix (.csv file) of training set with sample IDs on the rows (index) and genes of ineterest on the columns. Last column is the labels (0 for controls and 1 for cases)|
| --input_test   |Path to pEA matrix (.csv file) of testing set with samples on the rows (index) and genes of ineterest on the columns.Last column is the labels (0 for controls and 1 for cases)|
| --genes  | Path to a .csv file with genes of interest. Each gene should be on a separate row w/o a header |
| --savepath           | Path for output file |

## Command line example
```bash
#run RFE.py
python RFE.py --input_train Path/to/training_input_matrix.csv --input_test Path/to/testing_input_matrix.csv --savepath save/directory/ --genes Path/to/GenesofInterest/genes.csv
```

## Output
RFE repeatedly creates models and keeps aside the worst performing gene at each iteration until all the genes are exhausted. Six different models including Random Forest(RF), AdaBoost (AB), Support Vector Machine (SVM), Logistic Regression (LR), Elasticnet, Gradient Boosting (GB) are trained on the given training input. All genes get ranked using these different classifiers and the ranking are saved in the 6 different output files as "model_ranking.csv".
Accuracy of the case control separation will be tested on the given testing matrix at each iteration for each model. Finally, the accuracy of each model vs the # of top genes will be plotted and saved in the output path as "model_ACC.png".  
