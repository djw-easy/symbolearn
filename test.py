from sklearn.model_selection import train_test_split
from sklearn.utils.random import check_random_state
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import pandas as pd
import numpy as np
import jax

# 在导入其他JAX模块之前设置
jax.config.update('jax_platform_name', 'cpu')


from src.expression import Expression
from src.node import Variable, Operator, _operator_map
from src.symbolic_estimators import SymbolicRegressor, SymbolicClassifier




rng = check_random_state(0)
# Training samples
X_train = rng.uniform(1, 2, 100).reshape(50, 2)
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
    initial_constants=1,
    add_node=0.0,
    # stopping_criteria=0.0001,
    ncycles_per_iteration=380,
    should_optimize_constants=False,
    n_jobs=1, verbose=1, random_state=42)
sr_single.fit(X_train, y_train_single)
print("\nPareto Front:")
df = sr_single.get_hof()
print(df)




