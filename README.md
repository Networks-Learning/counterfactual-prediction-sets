# Designing Decision Support Systems Using Counterfactual Prediction Sets

This repository contains the ImageNet16H-PS dataset and the code used in the paper [Designing Decision Support Systems Using Counterfactual Prediction Sets](https://arxiv.org/abs/2306.03928), presented at the ICML 2023 AI & HCI Workshop (**Best Paper Award**) and published at ICML 2024.

## **ImageNet16H-PS Dataset**
The dataset as well as a more detailed description are under ```study_data/```.  

## **Install Dependencies**

Experiments ran on python 3.10.9. To install the required libraries run:

```pip install -r requirements.txt```


## **Code Structure**

### **Bandit Algorithms**
* ```algorithms/successive_elimination.py``` implements vanilla successive elimination, assumption-free counterfactual successive elimination and counterfactual successive elimination.
* ```algorithms/ucb.py``` implements vanilla $\texttt{UCB1}$, assumption-free counterfactual $\texttt{UCB1}$ and counterfactual $\texttt{UCB1}$.

### **Setup**
* ```config.py``` includes the configuration of the experimental setup.
* ```utils.py```  implements data splitting as well as helper functions.
* ```models/model.py``` implements the model used by all decision support systems.
* ```conformal_prediction.py``` implements all conformal predictors given a model and a fixed calibration set.

### **Sensitivity Analysis**
```robustness/``` includes the datasets with violations of interventional monotonicity, as well as the script to produce them:
* ```datasets/permutation_violations``` is the directory under which the datasets with violations are saved 
* ```create_violations.py``` creates datasets with monotonicity violations on different amounts (30%, 60%, 100%) of data

### **Scripts for Running Experiments**
```scripts/``` includes the following scripts to execute the algorithms:
* ```batch_run.py``` executes all bandit algorithms for 30 realizations given the same calibration set.
* ```run_bandit.py``` executes one realization of one algorithm given a calibration set. It can be also optionally used to compute the empirical expert success probability for each decision support system, i.e., for each $\alpha$ value, under the strict implementation, given a fixed calibration set.

### **Evaluation**
```scripts/``` includes the following scripts to evaluate the performance of experts under the strict and lenient implementation:
* ```test_other.py``` computes the empirical expert success probability for each decision support system, i.e., for each $\alpha$ value, under the lenient implementation, given a fixed calibration set.
* ```misplaced_trust_loss.py``` computes the number of experts predictions in which the prediction sets do not contain the true label and the experts succeed, and total number of expert predictions in which the experts misplace their trust for each $\alpha$ value under the lenient implementation given a fixed calibration set.

### **Plots**
```plotters/``` includes the following scripts to produce the plots in the paper:
* ```monotonicity.py``` produces the plots related to the empirical expert success probability per prediction set size for images of similar difficulty across all experts and across experts with the same level of competence.
* ```regret.py``` produces the empirical expected regret plot for all bandit algorithms.
* ```lenient.py``` produces the plots comparing the expert performance under the strict and lenient implementation for each $\alpha$ value.
* ```strict.py``` produces the plots showing the expert performance under the strict
implementation for each $\alpha$ value.

## **Running Instructions**

### **Run Experiments**
To execute all bandit algorithms for 30 realizations given a fixed calibration set run:

```python -m scripts.batch_run```

To execute a single bandit algorithm run:

```python -m scripts.run_bandit --alg ```<*algorithm*> ``` --seed_run ``` <*seed_run*> ```--cal_run ``` <*cal_run*>

where <*algorithm*> can be one of:
* ```SE```: vanilla successive elimination.
* ```SE_no_mon```: assumption-free counterfactual successive elimination.
* ```SE_ours```: counterfactual successive elimination.
* ```UCB```: vanilla $\texttt{UCB1}$.
* ```UCB_no_mon```: assumption-free counterfactual $\texttt{UCB1}$.
* ```UCB_ours```: counterfactual $\texttt{UCB1}$.

<*seed_run*> and <*cal_run*> can be any integer and fix the random seeds for  random procedures in the realization of the algorithm, and for randomly selecting the calibration set respectively. After the execution of each algorithm we save under ```results/```<*algorithm*>```/``` the arms that the algorithm pulled during the execution (required to compute the empirical expected regret).  

### **Evaluation and Plots** 
To produce the plots about the empirical success probability of experts per prediction set size for images of similar difficulty across all experts and across experts with the same level of competence run:

```python -m plotters.monotonicity```

To compute and plot the empirical expected regret for all bandit algorithms run:

```python -m scripts.eval_plot_regret```

To compute and plot the empirical expert success probability for each $\alpha$ under the strict and lenient implementation, as well as compute and plot the number of experts' predictions in which the prediction sets do not contain the true label and the experts succeed against the number of experts' predictions in which the prediction sets contain the true label and the experts predict from outside the prediction sets under the lenient implementation, run:


```python -m scripts.eval_plot_strict_vs_lenient```

All experiments use the ImageNet16H-PS dataset collected from the human subject study, which are in ```study_data/```. We include a detailed description and the license of the dataset in ```study_data/README.md```.


## **Running Sensitivity Analysis** 

### **Create Datasets with Violations**
To create the datasets with violations of interventional monotonicity for $p_v=\{0.3, 0.6, 1.0\}$
run:

```python -m robustness.create_violations```

### **Run Algorithms on Datasets with Violations** 
To execute all bandit algorithms for 30 realizations given a fixed calibration set
on datasets with violations on interventional monotonicity run:

```python -m scripts.batch_run --pv ```<*frac*>

where <*frac*> is the fraction of the data with violations, i.e., the $p_v$ value. In the paper, $p_v \in \{0.3, 0.6, 1.0\}$.

### **Evaluation and Plots**
To produce the plots about the empirical success probability of experts per prediction set size for images of similar difficulty across all experts for the datasets with interventional monotonicity violations, run:

```python -m  plotters.monotonicity --pv ```<*frac*>

where <*frac*> is the same as above.

To compute and plot the empirical expert success probability for each $\alpha$ under the strict implementation of our system for datasets with violations of interventional monotonicity run:

```python -m scripts.eval_plot_strict --pv ``` <*frac*>

where <*frac*> is the same as above. 

## **Citation**

If you use parts of the code/data in this repository for your own research purposes, please consider citing:

```
@article{straitouri2024designing,
  title={Designing Decision Support Systems Using Counterfactual Prediction Sets},
  author={Straitouri, Eleni and Gomez-Rodriguez, Manuel},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```
