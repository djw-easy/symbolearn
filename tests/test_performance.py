import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from typing import List, Dict, Union


from src.fitness import _fitness_map
from src.expression import Expression
from src.utils import check_random_state
from src.tree import SymbolicNode, PostOrderIter
from src.node import Constant, Variable, Operator, add2, mul2, sub2, div2, tanh1, sigmoid



# 1. 数据生成函数集合
def generate_data_for_expression(expression_type, n_samples=1000, noise_std=0.1, seed=42):
    """为不同表达式生成对应的测试数据"""
    np.random.seed(seed)
    X = np.array(np.random.randn(n_samples, 5), dtype=np.float32)
    
    if expression_type == "linear_combination":
        # y = 2.5*x0 + 1.3*x1 - 0.8
        y_true = X[:, 0] * 2.5 + X[:, 1] * 1.3 - 0.8
        true_params = np.array([2.5, 1.3, -0.8])
    elif expression_type == "with_interaction":
        # y = 1.8*x0 + 0.9*x1 + 0.5*x0*x1 - 1.2
        y_true = X[:, 0] * 1.8 + X[:, 1] * 0.9 + X[:, 0] * X[:, 1] * 0.5 - 1.2
        true_params = np.array([1.8, 0.9, 0.5, -1.2])
    elif expression_type == "nonlinear_simple":
        # y = tanh(1.5*x0 + 0.3) + 0.8*x1
        y_true = np.tanh(X[:, 0] * 1.5 + 0.3) + X[:, 1] * 0.8
        true_params = np.array([1.5, 0.3, 0.8])
    elif expression_type == "polynomial_with_tanh":
        # y = (1.2*x0 + 0.7*x1) * 0.9 + tanh(0.5*x2 - 0.2)
        linear_part = (X[:, 0] * 1.2 + X[:, 1] * 0.7) * 0.9
        nonlinear_part = np.tanh(X[:, 2] * 0.5 - 0.2)
        y_true = linear_part + nonlinear_part
        true_params = np.array([1.2, 0.7, 0.9, 0.5, -0.2])
    elif expression_type == "nested_division":
        # y = 2.0*(x0 + 0.5*x1) / (1.0 + sigmoid(1.5*x2 - 0.8))
        numerator = 2.0 * (X[:, 0] + 0.5 * X[:, 1])
        denominator = 1.0 + 1.0 / (1.0 + np.exp(-(1.5 * X[:, 2] - 0.8)))
        y_true = numerator / denominator
        true_params = np.array([2.0, 0.5, 1.0, 1.5, -0.8])
    elif expression_type == "transformation_chain":
        # y = tanh(1.1*(x0 - 0.4)) * sigmoid(0.9*x1 + 0.2) + 1.3*x2
        term1 = np.tanh(1.1 * (X[:, 0] - 0.4))
        term2 = 1.0 / (1.0 + np.exp(-(0.9 * X[:, 1] + 0.2)))  # sigmoid
        term3 = 1.3 * X[:, 2]
        y_true = term1 * term2 + term3
        true_params = np.array([1.1, 0.4, 0.9, 0.2, 1.3])
    else:
        raise ValueError(f"未知的表达式类型: {expression_type}")
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    return X, np.array(y, dtype=np.float32), y_true, true_params

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


def build_symbolic_tree(genes: List[Union[Constant, Variable, Operator]]) -> SymbolicNode:
    """
    将基因序列（后缀表达式）构建为符号树
    
    Args:
        genes: 基因序列，包含常数、变量和操作符
        
    Returns:
        构建好的符号树的根节点
    """
    stack = []
    
    for gene in genes:
        if isinstance(gene, (Constant, Variable)):
            # 叶节点：常数或变量
            node = SymbolicNode(node_content=gene)
            stack.append(node)
        
        elif isinstance(gene, Operator):
            # 操作符节点
            if len(stack) < gene.degree:
                raise ValueError(f"操作符 {gene.name} 需要 {gene.degree} 个参数，但栈中只有 {len(stack)} 个")
            
            # 弹出degree个子节点
            children = []
            for _ in range(gene.degree):
                children.append(stack.pop())
            children.reverse()  # 保持原来的顺序
            
            # 创建操作符节点
            op_node = SymbolicNode(node_content=gene)
            op_node.children = children
            stack.append(op_node)
        
        else:
            raise TypeError(f"不支持的基因类型: {type(gene)}")
    
    if len(stack) != 1:
        raise ValueError(f"表达式不完整，栈中剩余 {len(stack)} 个节点")
    
    return stack[0]


# 2. 定义优化函数
METHODS_WITH_EPS = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'SLSQP']

def optimize_constants_numpy(
    parent: Expression, X: np.ndarray, y: np.ndarray,
    optimizer_algorithm='L-BFGS-B', optimizer_nrestarts=3, 
    optimizer_iterations=10, random_state: np.random.RandomState = None
):
    random_state = check_random_state(random_state)
    # 获取初始常量
    initial_constants = np.array([
        node.node_content.value for node in PostOrderIter(parent.tree) 
            if isinstance(node.node_content, Constant)
    ])

    # 定义优化目标（使用预编译的梯度）
    def objective(constants_np: np.ndarray):
        # 计算损失（使用快速的NumPy执行）
        fitness = parent.fitness(X, y, constants_np)
        loss = -fitness if parent.metric.greater_is_better else fitness
        
        return loss

    # 多次重启优化
    best_loss = float('inf')
    best_constants = initial_constants.copy()
    
    for restart in range(optimizer_nrestarts):
        # 初始点
        if restart == 0:
            x0 = initial_constants.copy()
        else:
            noise_scale = 0.05 / np.sqrt(restart)
            noise = random_state.normal(0, noise_scale, size=len(initial_constants))
            constants_scale = np.abs(initial_constants) + 1e-6
            x0 = initial_constants + noise * constants_scale
        
        # 执行优化
        if optimizer_algorithm in METHODS_WITH_EPS:
            result = minimize(
                objective, x0, method=optimizer_algorithm, 
                options={'maxiter': optimizer_iterations, 'eps': 0.00001}
            )
        else:
            result = minimize(
                objective, x0,
                method=optimizer_algorithm, 
                options={'maxiter': optimizer_iterations}
            )
        # 更新最佳结果
        if result.fun < best_loss:
            best_loss = result.fun
            best_constants = result.x

    # 创建优化后的表达式
    optimized_expr = parent.update_constants(best_constants)
    final_fitness = -best_loss if parent.metric.greater_is_better else best_loss
    
    return optimized_expr, True, final_fitness



def optimize_constants_jax(
    expr: Expression, 
    X: np.ndarray, 
    y: np.ndarray,
    optimizer_algorithm='L-BFGS-B',
    optimizer_nrestarts=3,
    optimizer_iterations=10, 
    random_state: np.random.RandomState = None
):
    """
    混合策略的常量优化
    - 梯度计算：JAX（快速自动微分）
    - 适应度评估：NumPy（避免重复编译）
    """
    # 检查是否有常量
    if not (len(expr.constant_indices) > 0):
        return expr, False, expr.fitness(X, y)
    random_state = check_random_state(random_state)
    
    # 获取初始常量
    initial_constants = jnp.array([
        node.node_content.value for node in PostOrderIter(expr.tree) 
            if isinstance(node.node_content, Constant)
    ])
    
    # 预编译JAX梯度函数（只编译一次）
    grad_fn = expr._get_gradient_function()
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)
    
    # 定义优化目标（使用预编译的梯度）
    def objective_and_grad(constants_np):
        constants_jax = jnp.array(constants_np)
        
        # 计算梯度（使用预编译的函数）
        grad = grad_fn(constants_jax, X_jax, y_jax)
        
        # 计算损失（使用快速的NumPy执行）
        fitness = expr.fitness(X, y, constants_np)
        loss = -fitness if expr.metric.greater_is_better else fitness
        
        return float(loss), np.array(grad)
    
    # 多次重启优化
    best_loss = float('inf')
    best_constants = initial_constants.copy()
    
    for restart in range(optimizer_nrestarts):
        # 初始点
        if restart == 0:
            x0 = initial_constants.copy()
        else:
            noise_scale = 0.05 / np.sqrt(restart)
            noise = random_state.normal(0, noise_scale, size=len(initial_constants))
            constants_scale = np.abs(initial_constants) + 1e-6
            x0 = initial_constants + noise * constants_scale
        
        # 执行优化
        if optimizer_algorithm in METHODS_WITH_EPS:
            result = minimize(
                objective_and_grad, x0,
                method=optimizer_algorithm, jac=True,
                options={'maxiter': optimizer_iterations, 'eps': 0.00001}
            )
        else:
            result = minimize(
                objective_and_grad, x0,
                method=optimizer_algorithm, jac=True,
                options={'maxiter': optimizer_iterations}
            )
        
        # 更新最佳结果
        if result.fun < best_loss:
            best_loss = result.fun
            best_constants = result.x
    
    # 创建优化后的表达式
    optimized_expr = expr.update_constants(best_constants)
    
    # 计算最终适应度
    final_fitness = optimized_expr.fitness(X, y)
    
    return optimized_expr, True, final_fitness



n_samples = 3000


# 3. 测试表达式的执行速度
print("=" * 60)
print("测试表达式的执行速度")
# 预热
expr_name = list(raw_expressions.keys())[0]
X, y, y_true, true_params = generate_data_for_expression(
    expr_name, n_samples=n_samples, noise_std=0.05
)
tree = build_symbolic_tree(raw_expressions[expr_name]["genes"])
expression = Expression(tree=tree, metric=_fitness_map['mse'])
raw_fitness = expression.fitness(X, y)

for expr_name in raw_expressions.keys():
    print(f"测试表达式类型: {expr_name}")
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=n_samples, noise_std=0.05
    )
    start_time = time.time()
    n_iterations = 1000
    for i in range(n_iterations):
        tree = build_symbolic_tree(raw_expressions[expr_name]["genes"])
        expression = Expression(tree=tree, metric=_fitness_map['mse'])
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
print("测试能否进行优化 ----- JAX 优化器")
start_time = time.time()
for expr_name in raw_expressions.keys():
    tree = build_symbolic_tree(raw_expressions[expr_name]["genes"])
    expression = Expression(tree=tree, metric=_fitness_map['mse'])
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=n_samples, noise_std=0.05
    )
    print(f'模板表达式: {expression.update_constants(true_params)}')
    new_expr, _, _ = optimize_constants_jax(
        expression, X, y, optimizer_iterations=100, optimizer_nrestarts=1
    )
    print(f'优化表达式: {new_expr}')
    print(f'原始适应度: {expression.fitness(X, y)}')
    print(f'优化适应度: {new_expr.fitness(X, y)}')
end_time = time.time()
print(f"全部评估耗时: {end_time - start_time:.3f} 秒")
print("=" * 60)


print("\n" + "=" * 60)
print("测试能否进行优化 ----- Numpy 优化器")
start_time = time.time()
for expr_name in raw_expressions.keys():
    tree = build_symbolic_tree(raw_expressions[expr_name]["genes"])
    expression = Expression(tree=tree, metric=_fitness_map['mse'])
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=n_samples, noise_std=0.05
    )
    print(f'模板表达式: {expression.update_constants(true_params)}')
    new_expr, _, _ = optimize_constants_numpy(
        expression, X, y, optimizer_iterations=100, optimizer_nrestarts=1
    )
    print(f'优化表达式: {new_expr}')
    print(f'原始适应度: {expression.fitness(X, y)}')
    print(f'优化适应度: {new_expr.fitness(X, y)}')
end_time = time.time()
print(f"全部评估耗时: {end_time - start_time:.3f} 秒")
print("=" * 60)


# 5. 测试性能
print("\n" + "=" * 60)
print("测试性能 ----- JAX 优化器")
for expr_name in raw_expressions.keys():
    print(f"测试表达式类型: {expr_name}")
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=n_samples, noise_std=0.05
    )
    start_time = time.time()
    n_iterations = 100
    for i in range(n_iterations):
        tree = build_symbolic_tree(raw_expressions[expr_name]["genes"])
        expression = Expression(tree=tree, metric=_fitness_map['mse'])
        new_expr, _, _ = optimize_constants_jax(
            expression, X, y, optimizer_iterations=1000, optimizer_nrestarts=1
        )
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iterations * 1000
    throughput = n_iterations / (end_time - start_time)
    print(f"    评估 {n_iterations} 次耗时: {end_time - start_time:.3f} 秒")
    print(f"    平均每次: {avg_time:.3f} 毫秒")
    print(f"    吞吐量: {throughput:.0f} 次/秒")
print("=" * 60)


print("\n" + "=" * 60)
print("测试性能 ----- Numpy 优化器")
for expr_name in raw_expressions.keys():
    print(f"测试表达式类型: {expr_name}")
    X, y, y_true, true_params = generate_data_for_expression(
        expr_name, n_samples=n_samples, noise_std=0.05
    )
    start_time = time.time()
    n_iterations = 100
    for i in range(n_iterations):
        tree = build_symbolic_tree(raw_expressions[expr_name]["genes"])
        expression = Expression(tree=tree, metric=_fitness_map['mse'])
        new_expr, _, _ = optimize_constants_numpy(
            expression, X, y, optimizer_iterations=1000, optimizer_nrestarts=1
        )
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iterations * 1000
    throughput = n_iterations / (end_time - start_time)
    print(f"    评估 {n_iterations} 次耗时: {end_time - start_time:.3f} 秒")
    print(f"    平均每次: {avg_time:.3f} 毫秒")
    print(f"    吞吐量: {throughput:.0f} 次/秒")
print("=" * 60)

