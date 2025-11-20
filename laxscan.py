import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import List, Tuple
import numpy as np

# ============= 1. 定义操作码 =============
# 操作符
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3

# 特殊标记
TOKEN_VAR = 100   # 变量起始码: VAR_0=100, VAR_1=101, ...
TOKEN_CONST = 200 # 常量起始码: CONST_0=200, CONST_1=201, ...
TOKEN_PAD = -1    # 填充标记

# ============= 2. 核心求值器 =============
@jit
def evaluate_postfix(tokens, constants, X):
    """
    通用后缀表达式求值器 - 只编译一次，适用于所有表达式
    
    Args:
        tokens: (max_len,) 后缀表达式，例如 [100, 101, 0, 200, 2] 表示 (x0+x1)*c0
        constants: (n_constants,) 常量数组
        X: (batch_size, n_features) 输入数据
    
    Returns:
        (batch_size,) 计算结果
    """
    batch_size = X.shape[0]
    max_stack_size = len(tokens)  # 栈最大深度 = token数量
    
    # 初始化栈
    stack = jnp.zeros((max_stack_size, batch_size))
    stack_ptr = 0
    
    def process_token(carry, token):
        stack, stack_ptr = carry
        
        # 分支1: 变量 (token >= TOKEN_VAR and token < TOKEN_CONST)
        is_var = (token >= TOKEN_VAR) & (token < TOKEN_CONST)
        var_idx = token - TOKEN_VAR
        var_value = X[:, var_idx]
        
        # 分支2: 常量 (token >= TOKEN_CONST)
        is_const = token >= TOKEN_CONST
        const_idx = token - TOKEN_CONST
        const_value = jnp.where(
            const_idx < len(constants),
            constants[const_idx],
            0.0
        )
        
        # 分支3: 操作符 (token < TOKEN_VAR)
        is_op = token < TOKEN_VAR
        
        # 处理操作符
        def apply_operator(stack, stack_ptr, token):
            # 弹出两个操作数
            right = stack[stack_ptr - 1]
            left = stack[stack_ptr - 2]
            
            # 根据操作码计算
            result = jax.lax.switch(
                token,
                [
                    lambda l, r: l + r,  # ADD
                    lambda l, r: l - r,  # SUB
                    lambda l, r: l * r,  # MUL
                    lambda l, r: jnp.where(jnp.abs(r) > 1e-8, l / r, 1.0),  # DIV (保护)
                ],
                left, right
            )
            
            # 结果压栈
            stack = stack.at[stack_ptr - 2].set(result)
            return stack, stack_ptr - 1
        
        # 辅助函数：将值压入栈
        def push_to_stack(s, sp, value):
            return s.at[sp].set(value), sp + 1

        # 根据token类型选择操作
        stack, stack_ptr = jax.lax.switch(
            jnp.where(is_var, 0, jnp.where(is_const, 1, jnp.where(is_op, 2, 3))), # 0: var, 1: const, 2: op, 3: pad/invalid
            [
                lambda s, sp: push_to_stack(s, sp, var_value),  # 变量压栈
                lambda s, sp: push_to_stack(s, sp, const_value), # 常量压栈
                lambda s, sp: apply_operator(s, sp, token),     # 操作符
                lambda s, sp: (s, sp)                           # PAD或无效token，跳过
            ],
            stack, stack_ptr
        )
        
        return (stack, stack_ptr), None
    
    # 顺序处理所有token
    (stack, stack_ptr), _ = jax.lax.scan(process_token, (stack, stack_ptr), tokens)
    
    # 返回栈顶元素（最终结果）
    return stack[0]


# ============= 3. 损失和梯度计算 =============
@jit
def compute_loss_and_grad(tokens, constants, X, y):
    """
    计算MSE损失和常量的梯度
    
    Returns:
        loss: 标量
        grads: (n_constants,) 每个常量的梯度
    """
    def loss_fn(c):
        y_pred = evaluate_postfix(tokens, c, X)
        return jnp.mean((y_pred - y) ** 2)
    
    loss = loss_fn(constants)
    grads = grad(loss_fn)(constants)
    return loss, grads


# ============= 4. 批量处理 =============
@jit
def batch_evaluate(tokens_batch, constants_batch, X):
    """
    同时评估多个表达式
    
    Args:
        tokens_batch: (n_expr, max_len)
        constants_batch: (n_expr, max_constants)
        X: (batch_size, n_features)
    
    Returns:
        (n_expr, batch_size) 每个表达式在每个样本上的预测
    """
    return vmap(evaluate_postfix, in_axes=(0, 0, None))(
        tokens_batch, constants_batch, X
    )


@jit  
def batch_compute_loss_and_grad(tokens_batch, constants_batch, X, y):
    """
    同时计算多个表达式的损失和梯度
    
    Returns:
        losses: (n_expr,)
        grads: (n_expr, max_constants)
    """
    return vmap(compute_loss_and_grad, in_axes=(0, 0, None, None))(
        tokens_batch, constants_batch, X, y
    )


# ============= 5. 辅助工具 =============
class PostfixExpression:
    """后缀表达式包装类"""
    
    def __init__(self, tokens: List[int], constants: np.ndarray, max_len: int = 20):
        # 填充到固定长度
        self.tokens = np.array(tokens + [TOKEN_PAD] * (max_len - len(tokens)), dtype=np.int32)
        self.constants = jnp.array(constants, dtype=jnp.float32)
    
    def evaluate(self, X):
        """前向计算"""
        return evaluate_postfix(self.tokens, self.constants, X)
    
    def compute_loss_and_grad(self, X, y):
        """计算损失和梯度"""
        return compute_loss_and_grad(self.tokens, self.constants, X, y)
    
    def to_infix(self) -> str:
        """转换为中缀表达式（便于显示）"""
        stack = []
        op_names = {OP_ADD: '+', OP_SUB: '-', OP_MUL: '*', OP_DIV: '/'}
        
        for token in self.tokens:
            if token == TOKEN_PAD:
                break
            elif token >= TOKEN_CONST:
                const_idx = token - TOKEN_CONST
                stack.append(f'c{const_idx}')
            elif token >= TOKEN_VAR:
                var_idx = token - TOKEN_VAR
                stack.append(f'x{var_idx}')
            else:  # 操作符
                right = stack.pop()
                left = stack.pop()
                stack.append(f'({left} {op_names[token]} {right})')
        
        return stack[0] if stack else "empty"


# ============= 6. 使用示例 =============
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    X = jnp.array(np.random.randn(100, 5), dtype=jnp.float32)  # 100样本, 5特征
    y = X[:, 0] * 2.5 + X[:, 1] * 1.3 - 0.8  # 真实关系: y = 2.5*x0 + 1.3*x1 - 0.8
    
    print("=" * 60)
    print("示例1: 单个表达式优化")
    print("=" * 60)
    
    # 表达式: (x0 + x1) * c0 + c1
    # 后缀: [x0, x1, add, c0, mul, c1, add]
    expr = PostfixExpression(
        tokens=[100, 101, OP_ADD, 200, OP_MUL, 201, OP_ADD],
        constants=np.array([1.0, 0.0])  # 初始常量
    )
    
    print(f"表达式: {expr.to_infix()}")
    print(f"初始常量: {expr.constants}")
    
    # 优化常量
    learning_rate = 0.01
    for epoch in range(100):
        loss, grads = expr.compute_loss_and_grad(X, y)
        expr.constants = expr.constants - learning_rate * grads
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}, Constants = {expr.constants}")
    
    print(f"\n最终常量: {expr.constants}")
    print(f"理论值应接近: [1.9, -0.8] (因为 (x0+x1)*c0+c1 ≈ 2.5*x0+1.3*x1-0.8)")
    
    print("\n" + "=" * 60)
    print("示例2: 批量评估多个表达式（符号搜索场景）")
    print("=" * 60)
    
    # 创建3个候选表达式
    exprs_tokens = jnp.array([
        [100, 200, OP_MUL, 101, 201, OP_MUL, OP_ADD, TOKEN_PAD],  # x0*c0 + x1*c1
        [100, 101, OP_ADD, 200, OP_MUL, TOKEN_PAD, TOKEN_PAD, TOKEN_PAD],  # (x0+x1)*c0
        [100, 200, OP_MUL, 201, OP_ADD, TOKEN_PAD, TOKEN_PAD, TOKEN_PAD],  # x0*c0 + c1
    ])
    
    exprs_constants = jnp.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    
    # 一次性计算所有表达式的损失和梯度
    losses, grads = batch_compute_loss_and_grad(exprs_tokens, exprs_constants, X, y)
    
    print("表达式候选:")
    for i, (tokens, loss) in enumerate(zip(exprs_tokens, losses)):
        expr = PostfixExpression(tokens.tolist(), exprs_constants[i])
        print(f"  {i+1}. {expr.to_infix():30s} Loss = {loss:.6f}")
    
    best_idx = jnp.argmin(losses)
    print(f"\n最佳表达式: #{best_idx+1}")
    
    print("\n" + "=" * 60)
    print("性能测试: 编译一次，复用10000次")
    print("=" * 60)
    
    import time
    
    # 预热JIT编译
    _ = evaluate_postfix(expr.tokens, expr.constants, X)
    
    # 测试10000次评估
    n_iterations = 10000
    start = time.time()
    for _ in range(n_iterations):
        result = evaluate_postfix(expr.tokens, expr.constants, X)
        result.block_until_ready()  # 等待GPU完成
    end = time.time()
    
    print(f"评估 {n_iterations} 次耗时: {end - start:.3f}秒")
    print(f"平均每次: {(end - start) / n_iterations * 1000:.3f}毫秒")
    print(f"吞吐量: {n_iterations / (end - start):.0f} 次/秒")
