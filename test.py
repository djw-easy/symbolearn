from sklearn.model_selection import train_test_split
from sklearn.utils.random import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from scipy.io import loadmat
import jax.numpy as jnp
from numba import njit
import numpy as np
import jax
import os

# 在导入其他JAX模块之前设置
jax.config.update('jax_platform_name', 'cpu')


from src.fitness import Fitness
from src.symbolic_estimators import SymbolicRegressor, SymbolicClassifier, SymbolicTransformer


datasets_info = {
    'Houston': ['Houston', 'houston', 'Houston_gt', 'houston_gt', (349, 1905, 144), 15], 
    'Pavia University': ['PaviaU', 'paviaU', 'PaviaU_gt', 'paviaU_gt', (610, 340, 103), 9], 
    'Salinas': ['Salinas_corrected', 'salinas_corrected', 'Salinas_gt', 'salinas_gt', (514, 217, 224), 16], 
    'KSC': ['KSC', 'KSC', 'KSC_gt', 'KSC_gt', (512, 614, 176), 13],
    'Indian Pines': ['indian_pines_corrected', 'indian_pines_corrected', 'Indian_pines_gt', 'indian_pines_gt', (145, 145, 224), 16], 
    'Salinas-A': ['SalinasA_corrected', 'salinasA_corrected', 'SalinasA_gt', 'salinasA_gt', (86, 83, 224), 6], 
    'WHU_Hi_LongKou': ['WHU_Hi_LongKou', 'WHU_Hi_LongKou', 'WHU_Hi_LongKou_gt', 'WHU_Hi_LongKou_gt', (550, 400, 270), 9], 
    'WHU_Hi_HanChuan': ['WHU_Hi_HanChuan', 'WHU_Hi_HanChuan', 'WHU_Hi_HanChuan_gt', 'WHU_Hi_HanChuan_gt', (1217, 303, 274), 16]
}


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



# print(f"{'-'*23} Testing Single Expression for Symbolic Regression {'-'*23}")

# sr_single = SymbolicRegressor(
#     maxsize=15, 
#     niterations=10, 
#     populations=31,
#     population_size=27, 
#     use_constant=True,
#     use_aggregation=False,
#     ncycles_per_iteration=380,
#     should_optimize_constants=True,
#     should_optimize_aggregations=True,
#     n_jobs=16, verbose=1, random_state=42)
# sr_single.fit(X_train, y_train_single)
# print("\nPareto Front:")
# df = sr_single.get_hof()
# print(df)




# print(f"{'-'*34} Testing ExpressionSet for Symbolic Regression {'-'*34}")

# @jax.jit
# def mse_loss(y_true, y_pred):
#     y_pred = jnp.sum(y_pred, axis=1)
#     y_true = jnp.sum(y_true, axis=1)
#     return jnp.mean((y_true - y_pred) ** 2)
# mse_fitness = Fitness(mse_loss, greater_is_better=False, name="mse_n")

# sr_multi = SymbolicRegressor(
#     niterations=10, 
#     populations=31,
#     population_size=27, 
#     metric=mse_fitness,
#     use_constant=True,
#     maxsize=11, order=(2, 4),
#     use_aggregation=True,
#     # stopping_criteria=0.0001,
#     ncycles_per_iteration=380,
#     should_optimize_constants=True,
#     n_jobs=16, verbose=1, random_state=42)
# sr_multi.fit(X_train, y_train_multi)
# print("\nPareto Front:")
# df = sr_multi.get_hof()
# print(df)



print(f"{'-'*27} Testing Expression for Symbolic Transformer {'-'*27}")

rng = check_random_state(0)
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

from sklearn.linear_model import Ridge
est = Ridge()
est.fit(diabetes.data[:300, :], diabetes.target[:300])
print(est.score(diabetes.data[300:, :], diabetes.target[300:]))

operator_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                'abs', 'neg', 'inv', 'maximum', 'minimum']
st = SymbolicTransformer(
    maxsize=21, 
    niterations=10,
    populations=10,
    metric='pearson',
    population_size=27,
    use_constant=False,
    operators=operator_set,
    ncycles_per_iteration=90,
    n_jobs=8, verbose=1, random_state=42
)
X_new = st.fit_transform(diabetes.data[:300, :], diabetes.target[:300])



# if __name__ == '__main__':
#     print(f"{'-'*34} Testing ExpressionSet for Symbolic Classifier {'-'*34}")

#     selected_dataset = 'Pavia University'
#     dataset_info = datasets_info[selected_dataset]

#     # Load data
#     root_dir = os.getcwd()
#     X = loadmat(os.path.join(root_dir, f'data/hyperspectral/{selected_dataset}/{dataset_info[0]}.mat'))
#     X = X[dataset_info[1]]
#     y = loadmat(os.path.join(root_dir, f'data/hyperspectral/{selected_dataset}/{dataset_info[2]}.mat'))
#     y = y[dataset_info[3]]
#     X = X[y != 0, :]
#     y = y[y != 0]

#     # Standardize data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, train_size=200*dataset_info[-1], random_state=42, stratify=y
#     )
    
#     sc_classifier = SymbolicClassifier(
#         maxsize=17,
#         niterations=10,
#         populations=31,
#         population_size=27,
#         use_constant=True,
#         use_variable=True,
#         use_aggregation=False,
#         ncycles_per_iteration=38,
#         batching=True, batch_size=512,
#         should_optimize_constants=True,
#         should_optimize_aggregations=True,
#         n_jobs=16, verbose=1, random_state=42,
#         metric='cross_entropy', out_func='softmax',
#         aggregation_operators=('mean', 'min', 'max'),
#         operators=('+', '-', '*', 'sigmoid', 'tanh', 'softplus')
#     )
#     sc_classifier.fit(X_train, y_train)
    
#     # sc_classifier = SymbolicClassifier.from_file(
#     #     './data/Pavia University.csv',
#     #     maxsize=31, 
#     #     n_variables=103, 
#     #     classes=np.unique(y_train),
#     #     metric='focal_loss', out_func='softmax',
#     #     aggregation_operators=('mean', 'min', 'max'),
#     #     operators=('+', '*', 'sigmoid', 'tanh', 'softplus')
#     # )
#     # print(sc_classifier.get_best().expression.fitness(X_train, y_train-1))
#     print("TrainSet模型准确率:", sc_classifier.score(X_train, y_train))
#     print("TestSet模型准确率:", sc_classifier.score(X_test, y_test))
#     print("\nPareto Front:")
#     df = sc_classifier.get_hof()
#     print(df)

