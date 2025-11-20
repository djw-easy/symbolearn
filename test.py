from sklearn.model_selection import train_test_split
from sklearn.utils.random import check_random_state
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import pandas as pd
import numpy as np
import jax

# 在导入其他JAX模块之前设置
jax.config.update('jax_platform_name', 'cpu')


from src.fitness import Fitness
from src.symbolic_estimators import SymbolicRegressor, SymbolicClassifier




rng = check_random_state(0)
# Training samples
X_train = rng.uniform(1, 2, 5000).reshape(500, 10)
y_train_single = (
    X_train[:, 0]**2 
    - X_train[:, 1]**2 
    + X_train[:, 1] + 0.5
)
y_train_multi = np.stack(
    [
        X_train[:, 0]**2 - X_train[:, 1]**2, 
        X_train[:, 0]*X_train[:, 1] + X_train[:, 1] + 0.5
    ],
    axis=1
)



print(f"{'-'*23} Testing Single Expression for Symbolic Regression {'-'*23}")

sr_single = SymbolicRegressor(
    maxsize=15, 
    niterations=10, 
    populations=31,
    population_size=27, 
    use_constant=True,
    initial_constants=10,
    add_node=0.0, use_aggregation=True,
    # stopping_criteria=0.0001,
    ncycles_per_iteration=380,
    should_optimize_constants=True,
    n_jobs=1, verbose=1, random_state=42)
sr_single.fit(X_train, y_train_single)
print("\nPareto Front:")
df = sr_single.get_hof()
print(df)




# print(f"{'-'*23} Testing ExpressionSet for Symbolic Regression {'-'*23}")

# def mse_loss(y_true, y_pred, w):
#     y_pred = np.sum(y_pred, axis=1)
#     y_true = np.sum(y_true, axis=1)
#     return np.mean((y_true - y_pred) ** 2)
# mse_fitness = Fitness(mse_loss, greater_is_better=False)

# sr_multi = SymbolicRegressor(
#     niterations=10, 
#     populations=31,
#     population_size=27, 
#     metric=mse_fitness,
#     use_constant=True,
#     maxsize=9, order=(1, 4),
#     # stopping_criteria=0.0001,
#     ncycles_per_iteration=380,
#     # optimizer_algorithm='fast-gd',
#     n_jobs=1, verbose=1, random_state=42)
# sr_multi.fit(X_train, y_train_multi)
# print("\nPareto Front:")
# df = sr_multi.get_hof()
# print(df)

