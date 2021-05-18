![](https://raw.githubusercontent.com/MK-tech20/CGA2M_plus/main/images/cga2m_plus%2B.png) 
# CGA2M+ (Constraint GA2M plus)
We propose Constraint GA2M plus (CGA2M+), which we proposed. CGA2M+ is a modified version of GA2M to improve its interpretability and accuracy.
For more information, please read our paper.(coming soon!!) 
Mainly, CGA2M+ differs from GA2M in two respects.
1. introducing monotonic constraints
2. introducing higher-order interactions keeping the interpretability of the model
# Description of CGA2M+
Mainly, CGA2M+ differs from GA2M in two respects. We are using LightGBM as a shape function.

- introducing monotonic constraints
By adding monotonicity, we can improve the interpretability of our model. For example, we can make sure that "in the real estate market, as the number of rooms increases, the price decreases" does not happen. Human knowledge is needed to determine which features to enforce monotonicity on. The monotonicity constraint algorithm is implemented in LightGBM. This is a way to constrain the branches of a tree. For more details, please refer to the LightGBM implementation.

![](https://raw.githubusercontent.com/MK-tech20/CGA2M_plus/main/images/constraint.png)   

- introducing higher-order interactions keeping the interpretability of the model
GGA2M is unable to take into account higher-order interactions. Therefore, we introduce higher-order terms that are not interpretable. However, we devise a learning method so that the higher-order terms do not compromise the overall interpretability. Specifically, we train the higher-order terms as models that predict the residuals of the univariate terms and pairwise interaction terms. This allows most of the predictions to be explained by the interpretable first and second order terms. These residuals are then predicted by a higher-order term.

# Algorithm  
![](https://raw.githubusercontent.com/MK-tech20/CGA2M_plus/main/images/algorithm.png)
For more information, please read our paper. (coming soon!!) 
# Installation
You can get CGA2M+ from PyPI. Our project in PyPI is [here](https://pypi.org/project/cga2m-plus/).
```bash
pip install cga2m-plus
```

# Usage
For more detail, please read `examples/How_to_use_CGA2M+.ipynb`.
If it doesn't render at all in github, please click [here](https://kokes.github.io/nbviewer.js/viewer.html#aHR0cHM6Ly9naXRodWIuY29tL01LLXRlY2gyMC9DR0EyTV9wbHVzL2Jsb2IvbWFpbi9leGFtcGxlcy9Ib3dfdG9fdXNlX0NHQTJNJTJCLmlweW5i).
## Training

```python
cga2m = Constraint_GA2M(X_train,
                        y_train,
                        X_eval,
                        y_eval,
                        lgbm_params,
                        monotone_constraints = [0] * 6,
                        all_interaction_features = list(itertools.combinations(range(X_test.shape[1]), 2)))

cga2m.train(max_outer_iteration=20,backfitting_iteration=20,threshold=0.05)
cga2m.prune_and_retrain(threshold=0.05,backfitting_iteration=30)
cga2m.higher_order_train()
```
## Predict
```python
cga2m.predict(X_test,higher_mode=True)
```

## Visualize the effect of features on the target variables.
```python
plot_main(cga2m_no1,X_train)
```
![](https://raw.githubusercontent.com/MK-tech20/CGA2M_plus/main/images/plot_main.png) 

## Visualize (3d) the effect of pairs of features on the target variables
```python
plot_interaction(cga2m_no1,X_train,mode = "3d")
```
![](https://raw.githubusercontent.com/MK-tech20/CGA2M_plus/main/images/plot_pairs.png) 
## Feature importance
```python
show_importance(cga2m_no1,after_prune=True,higher_mode=True)
```
![](https://raw.githubusercontent.com/MK-tech20/CGA2M_plus/main/images/feature_importance.png) 
# License
MIT License
# Citation
You may use our package(CGA2M+) under MIT License. 
If you use this program in your research then please cite:

**CGA2M+ Package**  
```bash
M. Kuramata, A. Watanabe, K. Majima, H. Kiyohara, M. Iwata, K. Kondo, K. Nakata, 
CGA2M+ Package. (2021) [Online].
Available: https://github.com/MK-tech20/CGA2M_plus
```

**CGA2M+ Paper**  
The paper has not been published yet. 

# Reference
[1] Friedman, J. H. 2001, Greedy function approximation: a gradient boosting machine, Annals of statistics, 1189-1232, doi: 10.1214/aos/1013203451. Available online: May 02, 2021

[2] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... Liu, T. Y. 2017. Lightgbm: A highly efficient gradient boosting decision tree, Advances in neural information processing systems(NIPS’17), Long Beach California , 4-9 December, pp. 3146-3154.

[3] Nelder, J. A., Wedderburn, R. W. 1972. Generalized linear models, Journal of the Royal Statistical Society: Series A (General), 135(3), 370-384, doi: 10.2307/2344614, Available online: May 02, 2021

[4] Hastie, T. J., Tibshirani, R. J. 1990. Generalized additive models (Vol. 43), CRC press, doi: 10.1214/ss/1177013604. Available online: May 02, 2021

[5] Lou, Y., Caruana, R., Gehrke, J., Hooker, G. 2013, August. Accurate intelligible models with pairwise interactions, Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining(KDD’13), Chicago Illinois, United States of America, 11-14 August, pp. 623-631.

[6] “GitHub - microsoft/LightGBM” [Online]. Available: https://github.com/microsoft/LightGBM (Accessed: May 02, 2021)

[7] “scikit-learn: machine learning in Python — scikit-learn 0.24.2 documentation” [Online]. Available: https://scikit-learn.org/stable/ (Accessed May 02, 2021)
