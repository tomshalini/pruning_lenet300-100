# pruning_lenet300-100
Pruned LeNet300-100 using Iterative pruning based on https://arxiv.org/abs/1506.02626 

Steps to follow:
1. create conda enviornment with tensorflow>2.0 and matplotlib.


2. Run main.py and update quality_parameter value to change pruning theshold.
   (Iterative training (prune+retrain) in single iteration and run until we get pruning accuracy >= original model accuracy.
    Threshold is calculated based on standard deviation of weights and sensitivity/quality factor).

3. Results are uploaded in results folder. 
   original model, pruned model, and retarin model accuracy & compression rate during iteraive pruning.


![Pruning Results]
![Pruning Results](https://github.com/tomshalini/pruning_lenet300-100/blob/master/results/result1.png)
![Pruning Results](https://github.com/tomshalini/pruning_lenet300-100/blob/master/results/result2.png)


