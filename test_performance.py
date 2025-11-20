import jax
import time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import optax

from src.fitness import _fitness_map
from src.expression import Expression
from src.utils import check_random_state
from src.node import Constant, Variable, add2, mul2, sub2, div2, tanh1, sigmoid, softplus



# 1. 数据生成函数集合
def generate_data_for_expression(expression_type, n_samples=1000, noise_std=0.1, seed=42):
    """为不同表达式生成对应的测试数据"""
    np.random.seed(seed)
    X = jnp.array(np.random.randn(n_samples, 5), dtype=jnp.float32)
    
    if expression_type == "linear_combination":
        # y = 2.5*x0 + 1.3*x1 - 0.8
        y_true = X[:, 0] * 2.5 + X[:, 1] * 1.3 - 0.8
        true_params = jnp.array([2.5, 1.3, -0.8])
        
    elif expression_type == "with_interaction":
        # y = 1.8*x0 + 0.9*x1 + 0.5*x0*x1 - 1.2
        y_true = X[:, 0] * 1.8 + X[:, 1] * 0.9 + X[:, 0] * X[:, 1] * 0.5 - 1.2
        true_params = jnp.array([1.8, 0.9, 0.5, -1.2])
        
    elif expression_type == "nonlinear_simple":
        # y = tanh(1.5*x0 + 0.3) + 0.8*x1
        y_true = jnp.tanh(X[:, 0] * 1.5 + 0.3) + X[:, 1] * 0.8
        true_params = jnp.array([1.5, 0.3, 0.8])
        
    elif expression_type == "polynomial_with_tanh":
        # y = (1.2*x0 + 0.7*x1) * 0.9 + tanh(0.5*x2 - 0.2)
        linear_part = (X[:, 0] * 1.2 + X[:, 1] * 0.7) * 0.9
        nonlinear_part = jnp.tanh(X[:, 2] * 0.5 - 0.2)
        y_true = linear_part + nonlinear_part
        true_params = jnp.array([1.2, 0.7, 0.9, 0.5, -0.2])
        
    elif expression_type == "nested_division":
        # y = 2.0*(x0 + 0.5*x1) / (1.0 + sigmoid(1.5*x2 - 0.8))
        numerator = 2.0 * (X[:, 0] + 0.5 * X[:, 1])
        denominator = 1.0 + 1.0 / (1.0 + jnp.exp(-(1.5 * X[:, 2] - 0.8)))
        y_true = numerator / denominator
        true_params = jnp.array([2.0, 0.5, 1.0, 1.5, -0.8])
        
    elif expression_type == "multi_var_combination":
        # y = softplus(0.8*x0) + 1.2*x1*x2 - 0.6/(x3 + 0.3)
        term1 = jnp.log(1.0 + jnp.exp(0.8 * X[:, 0]))  # softplus
        term2 = 1.2 * X[:, 1] * X[:, 2]
        term3 = 0.6 / (X[:, 3] + 0.3)
        y_true = term1 + term2 - term3
        true_params = jnp.array([0.8, 1.2, 0.6, 0.3])
        
    elif expression_type == "transformation_chain":
        # y = tanh(1.1*(x0 - 0.4)) * sigmoid(0.9*x1 + 0.2) + 1.3*x2
        term1 = jnp.tanh(1.1 * (X[:, 0] - 0.4))
        term2 = 1.0 / (1.0 + jnp.exp(-(0.9 * X[:, 1] + 0.2)))  # sigmoid
        term3 = 1.3 * X[:, 2]
        y_true = term1 * term2 + term3
        true_params = jnp.array([1.1, 0.4, 0.9, 0.2, 1.3])
        
    else:
        raise ValueError(f"未知的表达式类型: {expression_type}")
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    return X, jnp.array(y, dtype=jnp.float32), y_true, true_params

# 2. 表达式定义
raw_expressions = {
    # 简单表达式
    "linear_combination": {
        "genes": [
            Constant(0), Variable(0), mul2,
            Constant(1), Variable(1), mul2,
            add2,
            Constant(2), add2
        ],
        "description": "线性组合: c0*x0 + c1*x1 + c2"
    },
    
    "with_interaction": {
        "genes": [
            Constant(0), Variable(0), mul2,
            Constant(1), Variable(1), mul2,
            add2,
            Variable(0), Variable(1), mul2,
            Constant(2), mul2,
            add2,
            Constant(3), add2
        ],
        "description": "带交互项: c0*x0 + c1*x1 + c2*x0*x1 + c3"
    },
    
    "nonlinear_simple": {
        "genes": [
            Constant(0), Variable(0), mul2,
            Constant(1), add2,
            tanh1,
            Constant(2), Variable(1), mul2,
            add2
        ],
        "description": "非线性变换: tanh(c0*x0 + c1) + c2*x1"
    },
    
    # 复杂表达式
    "polynomial_with_tanh": {
        "genes": [
            Constant(0), Variable(0), mul2,
            Constant(1), Variable(1), mul2,
            add2,
            Constant(2), mul2,
            Constant(3), Variable(2), mul2,
            Constant(4), add2,
            tanh1,
            add2
        ],
        "description": "多项式+tanh: (c0*x0 + c1*x1)*c2 + tanh(c3*x2 + c4)"
    },
    
    "nested_division": {
        "genes": [
            Variable(0),
            Constant(1), Variable(1), mul2,
            add2,
            Constant(0), mul2,
            Constant(3), Variable(2), mul2,
            Constant(4), sub2,
            sigmoid,
            Constant(2), add2,
            div2
        ],
        "description": "嵌套除法: c0*(x0 + c1*x1) / (c2 + sigmoid(c3*x2 - c4))"
    },
    
    "transformation_chain": {
        "genes": [
            Variable(0), Constant(1), sub2,
            Constant(0), mul2,
            tanh1,
            Constant(2), Variable(1), mul2,
            Constant(3), add2,
            sigmoid,
            mul2,
            Constant(4), Variable(2), mul2,
            add2
        ],
        "description": "变换链: tanh(c0*(x0-c1)) * sigmoid(c2*x1+c3) + c4*x2"
    }
}


# 2. 定义优化函数
METHODS_WITH_EPS = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP']
def optimize_constants(parent: Expression, X, y, 
                       optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, 
                       optimizer_iterations=10, random_state: np.random.RandomState = None):
    random_state = check_random_state(random_state)
    # 1. 收集所有常量的索引（一次遍历）
    constant_indices = [i for i, gene in enumerate(parent.genes) 
                        if isinstance(gene, Constant)]
    
    if not constant_indices:
        return None, False, np.nan
    
    # 2. 提取初始常量值
    initial_constants = jnp.array([parent.genes[i].value for i in constant_indices])
    
    # 3. 定义目标函数
    @jax.jit
    def objective(constants: jnp.ndarray):
        fitness = parent._fitness_for_grad(X, y, constants)
        loss = -fitness if parent.metric.greater_is_better else fitness
        return loss

    # JIT编译梯度计算
    grad_fn = jax.jit(jax.grad(objective))

    def scipy_wrapper(x):
        x_jax = jnp.array(x)
        return float(objective(x_jax)), np.array(grad_fn(x_jax))
    
    # JAX自动计算梯度
    grad_fn = jax.jit(jax.grad(objective))
    
    # 4. 多次重启优化（寻找全局最优）
    best_loss = objective(initial_constants)
    best_constants = initial_constants.copy()
    
    for restart in range(optimizer_nrestarts):
        # 第一次使用原始值，后续添加噪声
        if restart == 0:
            x0 = initial_constants.copy()
        else:
            # 噪声强度递减（避免后期扰动过大）
            noise_scale = 0.05 / np.sqrt(restart)
            # restart=1: 5%, restart=2: 3.5%, restart=3: 2.9%
            noise = random_state.normal(0, noise_scale, size=len(initial_constants))
            constants_scale = np.abs(initial_constants) + 1e-6  # 处理零值
            x0 = initial_constants + noise * constants_scale
        
        # 执行优化
        if optimizer_algorithm in METHODS_WITH_EPS:
            result = minimize(
                scipy_wrapper, x0,
                method=optimizer_algorithm, jac=True,
                options={'maxiter': optimizer_iterations, 'eps': 0.00001}
            )
        else:
            result = minimize(
                scipy_wrapper, x0,
                method=optimizer_algorithm, jac=True,
                options={'maxiter': optimizer_iterations}
            )
        
        # 更新最佳结果
        if result.fun < best_loss:
            best_loss = result.fun
            best_constants = result.x
    
    # 5. 应用最佳常量
    new_genes = parent.genes.copy()
    for idx, const_idx in enumerate(constant_indices):
        new_genes[const_idx] = Constant(best_constants[idx])
    new_expr = Expression(genes=new_genes, metric=parent.metric)
    
    # 6. 更新适应度
    raw_fitness = -best_loss if new_expr.metric.greater_is_better else best_loss
    
    return new_expr, True, raw_fitness

def optimize_constants_jax(parent: Expression, X, y,
                           learning_rate=0.1, optimizer_iterations=50,
                           optimizer_nrestarts=3, random_state: np.random.RandomState = None):
    """
    使用 JAX 原生优化器 (optax) 加速常量优化。
    """
    random_state = check_random_state(random_state)
    
    # 1. 收集常量信息
    constant_indices = [i for i, gene in enumerate(parent.genes) if isinstance(gene, Constant)]
    if not constant_indices:
        return parent, False, np.nan

    initial_constants = jnp.array([parent.genes[i].value for i in constant_indices])

    # 2. 定义 JAX 原生的目标函数
    def objective(constants: jnp.ndarray):
        fitness = parent._fitness_for_grad(X, y, constants)
        loss = -fitness if parent.metric.greater_is_better else fitness
        return loss

    # 3. 定义并 JIT 编译整个优化步骤
    # 使用 value_and_grad 可以同时计算损失和梯度，更高效
    grad_fn = jax.value_and_grad(objective)

    # 选择一个 optax 优化器，Adam 是一个很好的默认选择
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def optimization_step(params, opt_state):
        """执行单步优化，这个函数将被 JIT 编译"""
        loss, grads = grad_fn(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # 4. 多次重启优化
    best_loss = jnp.inf
    best_constants = initial_constants

    for restart in range(optimizer_nrestarts):
        # 初始化参数和优化器状态
        if restart == 0:
            current_constants = initial_constants
        else:
            noise_scale = 0.05 / np.sqrt(restart)
            noise = random_state.normal(0, noise_scale, size=len(initial_constants))
            constants_scale = np.abs(initial_constants) + 1e-6
            current_constants = initial_constants + noise * constants_scale
            current_constants = jnp.array(current_constants)

        opt_state = optimizer.init(current_constants)

        # 运行优化循环
        # 注意：这里的 for 循环在 Python 中运行，但每次循环调用的是一个
        # 已经 JIT 编译好的高效函数 optimization_step。
        for _ in range(optimizer_iterations):
            current_constants, opt_state, _ = optimization_step(current_constants, opt_state)

        # 评估本次重启的结果
        final_loss = objective(current_constants)
        if final_loss < best_loss:
            best_loss = final_loss
            best_constants = current_constants

    # 5. 应用最佳常量
    new_genes = parent.genes.copy()
    for idx, const_idx in enumerate(constant_indices):
        new_genes[const_idx] = Constant(best_constants[idx])
    new_expr = Expression(genes=new_genes, metric=parent.metric)

    # 6. 计算最终适应度
    raw_fitness = -best_loss if new_expr.metric.greater_is_better else best_loss

    return new_expr, True, float(raw_fitness)


def expr_with_constants(expr: Expression, constants: np.ndarray):
    constant_indices = [i for i, gene in enumerate(expr.genes) 
                        if isinstance(gene, Constant)]
    new_genes = expr.genes.copy()
    for idx, const_idx in enumerate(constant_indices):
        new_genes[const_idx] = Constant(constants[idx])
    new_expr = Expression(genes=new_genes, metric=expr.metric)
    return new_expr


# 3. 测试表达式的执行速度
print("=" * 60)
print("测试表达式的执行速度")
for expr_name in raw_expressions.keys():
    print(f"测试表达式类型: {expr_name}")
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=1000, noise_std=0.05
    )
    start_time = time.time()
    n_iterations = 1000
    for i in range(n_iterations):
        expression = Expression(genes=raw_expressions[expr_name]["genes"], metric=_fitness_map['mse'])
        raw_fitness = expression.fitness(X, y)
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iterations * 1000
    throughput = n_iterations / (end_time - start_time)
    print(f"    评估 {n_iterations} 次耗时: {end_time - start_time:.3f} 秒")
    print(f"    平均每次: {avg_time:.3f} 毫秒")
    print(f"    吞吐量: {throughput:.0f} 次/秒")
print("=" * 60)


# 4. 测试能否进行优化
print("\n" + "=" * 60)
print("测试能否进行优化")
start_time = time.time()
for expr_name in raw_expressions.keys():
    expression = Expression(genes=raw_expressions[expr_name]["genes"], metric=_fitness_map['mse'])
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=1000, noise_std=0.05
    )
    print(f'模板表达式: {expr_with_constants(expression, true_params)}')
    new_expr, _, _ = optimize_constants(expression, X, y)
    print(f'优化表达式: {new_expr}')
end_time = time.time()
print(f"全部评估耗时: {end_time - start_time:.3f} 秒")
print("=" * 60)


# 5. 测试性能
print("\n" + "=" * 60)
print("测试性能")
for expr_name in raw_expressions.keys():
    print(f"测试表达式类型: {expr_name}")
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=1000, noise_std=0.05
    )
    start_time = time.time()
    n_iterations = 100
    for i in range(n_iterations):
        expression = Expression(genes=raw_expressions[expr_name]["genes"], metric=_fitness_map['mse'])
        new_expr, _, _ = optimize_constants_jax(expression, X, y, optimizer_iterations=50)
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iterations * 1000
    throughput = n_iterations / (end_time - start_time)
    print(f"    评估 {n_iterations} 次耗时: {end_time - start_time:.3f} 秒")
    print(f"    平均每次: {avg_time:.3f} 毫秒")
    print(f"    吞吐量: {throughput:.0f} 次/秒")
print("=" * 60)




