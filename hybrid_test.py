import jax
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from src.node_jax import Operator, Constant, Variable, NodeContent, DynamicAggregation


class HybridExpression:
    """
    混合策略表达式：
    - 日常执行用NumPy（快速评估）
    - 常量优化用JAX（自动微分）
    """
    def __init__(self, genes: List[NodeContent], out_func=None, metric=None):
        self.genes = genes
        self.metric = metric
        self.out_func = out_func
        
        # 预分析表达式结构
        self._constant_indices = [i for i, g in enumerate(genes) if isinstance(g, Constant)]
        self._has_constants = len(self._constant_indices) > 0
        
        # 懒编译梯度函数（只在需要时编译一次）
        self._grad_fn_compiled = None
    
    @property
    def size(self):
        return len(self.genes)
    
    def copy(self):
        return HybridExpression(self.genes.copy(), self.out_func, self.metric)
    
    # ============ NumPy快速执行 ============
    def execute(self, X: np.ndarray) -> np.ndarray:
        """
        使用NumPy执行（快速路径）
        避免JAX编译开销
        """
        X_np = np.asarray(X)  # 确保是NumPy数组
        stack = []
        
        for gene in self.genes:
            if gene.degree == 0:
                # Variable, Constant, DynamicAggregation
                result = gene(X_np)
                # 转换为NumPy（如果是JAX数组）
                if hasattr(result, 'device'):
                    result = np.array(result)
                stack.append(result)
            else:
                # Operator
                operands = [stack.pop() for _ in range(gene.degree)]
                operands.reverse()
                
                # 使用NumPy版本的函数
                result = self._apply_operator_numpy(gene, operands)
                stack.append(result)
        
        result = stack[0]
        
        # 处理标量
        if np.isscalar(result) or result.ndim == 0:
            result = np.full(X_np.shape[0], result)
        
        # 应用输出函数
        if self.out_func is not None:
            result = self._apply_operator_numpy(self.out_func, [result])
        
        return result
    
    def _apply_operator_numpy(self, operator, operands):
        """将JAX操作转换为NumPy操作"""
        # 直接调用operator，它内部使用的jnp函数会自动fallback到numpy
        # 但为了性能，可以显式转换
        operands_np = [np.asarray(op) for op in operands]
        
        # 调用operator（JAX函数会自动在NumPy数组上工作）
        result = operator(*operands_np)
        
        return np.asarray(result)
    
    def fitness(self, X: np.ndarray, y: np.ndarray) -> float:
        """快速适应度评估（NumPy）"""
        y_pred = self.execute(X)
        # 确保转换为NumPy
        y_np = np.asarray(y)
        y_pred_np = np.asarray(y_pred)
        
        # 调用metric（可能是JAX函数，但在NumPy数组上也能工作）
        raw_fitness = float(self.metric(y_np, y_pred_np))
        return raw_fitness
    
    # ============ JAX梯度计算（仅用于常量优化） ============
    def _build_jax_executable(self):
        """
        构建JAX可执行版本（仅在需要梯度时调用）
        关键优化：使用静态结构避免动态循环
        """
        genes = self.genes
        constant_indices = self._constant_indices
        out_func = self.out_func
        
        # 提取常量索引映射
        const_idx_map = {idx: i for i, idx in enumerate(constant_indices)}
        
        def _execute_with_constants(X_jax, constants):
            """JAX执行函数（可微分）"""
            stack = []
            const_counter = 0
            
            for i, gene in enumerate(genes):
                if gene.degree == 0:
                    if i in const_idx_map:
                        # 使用可优化的常量
                        stack.append(constants[const_counter])
                        const_counter += 1
                    else:
                        # Variable或DynamicAggregation
                        stack.append(gene(X_jax))
                else:
                    # Operator
                    operands = [stack.pop() for _ in range(gene.degree)]
                    operands.reverse()
                    result = gene(*operands)
                    stack.append(result)
            
            result = stack[0]
            
            # 处理标量
            if result.ndim == 0:
                result = jnp.full(X_jax.shape[0], result)
            
            # 应用输出函数
            if out_func is not None:
                result = out_func(result)
            
            return result
        
        return _execute_with_constants
    
    def _get_gradient_function(self):
        """懒编译梯度函数"""
        if self._grad_fn_compiled is None:
            executable = self._build_jax_executable()
            
            def loss_fn(constants, X_jax, y_jax):
                """损失函数（用于计算梯度）"""
                y_pred = executable(X_jax, constants)
                loss = self.metric(y_jax, y_pred)
                # 如果是最大化指标，取负数
                if self.metric.greater_is_better:
                    loss = -loss
                return loss
            
            # JIT编译梯度函数
            self._grad_fn_compiled = jax.jit(jax.grad(loss_fn, argnums=0))
        
        return self._grad_fn_compiled
    
    def compute_constant_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算常量的梯度（使用JAX）
        返回：(梯度数组, 当前损失值)
        """
        if not self._has_constants:
            return None, None
        
        # 转换为JAX数组
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # 提取当前常量值
        current_constants = jnp.array([
            self.genes[idx].value for idx in self._constant_indices
        ])
        
        # 计算梯度
        grad_fn = self._get_gradient_function()
        gradients = grad_fn(current_constants, X_jax, y_jax)
        
        # 计算当前损失
        executable = self._build_jax_executable()
        y_pred = executable(X_jax, current_constants)
        loss = float(self.metric(y_jax, y_pred))
        if self.metric.greater_is_better:
            loss = -loss
        
        return np.array(gradients), loss
    
    def update_constants(self, new_values: np.ndarray):
        """更新常量值"""
        if len(new_values) != len(self._constant_indices):
            raise ValueError(f"Expected {len(self._constant_indices)} values, got {len(new_values)}")
        
        new_genes = self.genes.copy()
        for i, idx in enumerate(self._constant_indices):
            new_genes[idx] = Constant(float(new_values[i]))
        
        return HybridExpression(new_genes, self.out_func, self.metric)


# ============ 优化后的常量优化函数 ============

def optimize_constants_hybrid(
    expr: HybridExpression, 
    X: np.ndarray, 
    y: np.ndarray,
    optimizer_algorithm='L-BFGS-B',
    optimizer_nrestarts=3,
    optimizer_iterations=10,
    random_state=None
):
    """
    混合策略的常量优化
    - 梯度计算：JAX（快速自动微分）
    - 适应度评估：NumPy（避免重复编译）
    """
    from scipy.optimize import minimize
    
    if random_state is None:
        random_state = np.random.RandomState()
    
    # 检查是否有常量
    if not expr._has_constants:
        return expr, False, expr.fitness(X, y)
    
    # 获取初始常量
    initial_constants = np.array([
        expr.genes[idx].value for idx in expr._constant_indices
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
        temp_expr = expr.update_constants(constants_np)
        fitness = temp_expr.fitness(X, y)
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
        result = minimize(
            objective_and_grad,
            x0,
            method=optimizer_algorithm,
            jac=True,
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


# ============ 性能对比测试 ============

def benchmark_comparison():
    """比较不同实现的性能"""
    import time
    from src.node_jax import Variable, Constant, add2, mul2
    from src.fitness import _fitness_map
    
    # 创建测试表达式: (x1 + 2.5) * x2
    genes = [
        Variable(0),
        Constant(2.5),
        add2,
        Variable(1),
        mul2
    ]
    
    metric = _fitness_map['mse']
    expr = HybridExpression(genes, metric=metric)
    
    # 测试数据
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = np.random.randn(n_samples)
    
    print("=" * 60)
    print("性能对比测试")
    print("=" * 60)
    
    # 1. NumPy执行测试
    print("\n1. NumPy执行（日常评估）")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = expr.execute(X)
        times.append(time.perf_counter() - start)
    print(f"   平均时间: {np.mean(times)*1000:.3f}ms")
    print(f"   标准差: {np.std(times)*1000:.3f}ms")
    
    # 2. 梯度计算测试
    print("\n2. JAX梯度计算（常量优化）")
    # 首次调用（包含编译）
    start = time.perf_counter()
    grad, loss = expr.compute_constant_gradient(X, y)
    first_time = time.perf_counter() - start
    print(f"   首次调用（含编译）: {first_time*1000:.1f}ms")
    
    # 后续调用（复用编译）
    times = []
    for _ in range(10):
        start = time.perf_counter()
        grad, loss = expr.compute_constant_gradient(X, y)
        times.append(time.perf_counter() - start)
    print(f"   后续调用平均: {np.mean(times)*1000:.3f}ms")
    
    # 3. 完整优化测试
    print("\n3. 完整常量优化")
    start = time.perf_counter()
    opt_expr, success, fitness = optimize_constants_hybrid(
        expr, X, y, 
        optimizer_nrestarts=3,
        optimizer_iterations=10
    )
    opt_time = time.perf_counter() - start
    print(f"   总时间: {opt_time*1000:.1f}ms")
    print(f"   优化前适应度: {expr.fitness(X, y):.6f}")
    print(f"   优化后适应度: {fitness:.6f}")
    
    print("\n" + "=" * 60)
    print("结论：")
    print("- 日常执行：NumPy快速路径，无编译开销")
    print("- 常量优化：JAX梯度计算，只编译一次梯度函数")
    print("- 最佳平衡：混合策略实现")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_comparison()