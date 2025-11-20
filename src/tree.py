from typing import Union, Optional, List, Dict
import numpy as np


from src.utils import check_random_state



def get_integer_partitions(n: int, k: int):
    """
    一个生成器，用于将整数 n 划分为 k 个正整数部分。
    """
    if k == 1:
        if n > 0:
            yield (n,)
        return

    for i in range(1, n - (k - 1) + 1):
        for partition in get_integer_partitions(n - i, k - 1):
            yield (i,) + partition


def count_trees(
    size: int,
    degrees: List[int],
    memo: Dict[int, int]=None
) -> int:
    """
    计算大小恰好为 'size' 的树有多少种可能的组合。
    """
    if memo is None:
        memo = {}
    
    if size in memo:
        return memo[size]
    
    if size <= 0:
        return 0

    if size == 1:
        count = 1 if 0 in degrees else 0
        memo[1] = count
        return count

    total_count = 0
    nodes_for_children = size - 1

    for degree in degrees:
        if degree == 0 or nodes_for_children < degree:
            continue

        for partition in get_integer_partitions(nodes_for_children, degree):
            combinations_for_partition = 1
            possible = True
            for child_size in partition:
                child_count = count_trees(child_size, degrees, memo)
                if child_count == 0:
                    possible = False
                    break
                combinations_for_partition *= child_count
            
            if possible:
                total_count += combinations_for_partition
    
    memo[size] = total_count
    return total_count


def generate_random_tree(
    size: int,
    degrees: List[int],
    count_memo: Optional[Dict[int, int]] = None, 
    random_state: Optional[np.random.RandomState] = None
) -> Optional[List[int]]:
    """
    (优化版) 随机返回一个满足条件的树 (后缀 degree 列表)。
    """
    if count_memo is None:
        count_memo = {}
    random_state = check_random_state(random_state)
    
    if count_trees(size, degrees, count_memo) == 0:
        return None

    if size == 1:
        return [0]

    nodes_for_children = size - 1

    # === 1. 加权随机选择 degree ===
    degree_choices = []
    for degree in sorted(degrees):
        if degree == 0 or nodes_for_children < degree:
            continue
        
        count_for_this_degree = 0
        for partition in get_integer_partitions(nodes_for_children, degree):
            combinations_for_partition = 1
            possible = True
            for child_size in partition:
                child_count = count_trees(child_size, degrees, count_memo)
                if child_count == 0:
                    possible = False
                    break
                combinations_for_partition *= child_count
            if possible:
                count_for_this_degree += combinations_for_partition
        
        if count_for_this_degree > 0:
            degree_choices.append({'degree': degree, 'weight': count_for_this_degree})

    population_degrees = [c['degree'] for c in degree_choices]
    weights_degrees = np.array([c['weight'] for c in degree_choices], dtype=np.int64)
    probabilities_degrees = weights_degrees / np.sum(weights_degrees)
    chosen_degree = random_state.choice(population_degrees, p=probabilities_degrees)

    # === 2. 针对已选定的 degree，加权随机选择 Partition ===
    partition_choices = []
    for partition in get_integer_partitions(nodes_for_children, chosen_degree):
        combinations_for_partition = 1
        possible = True
        for child_size in partition:
            child_count = count_trees(child_size, degrees, count_memo)
            if child_count == 0:
                possible = False
                break
            combinations_for_partition *= child_count
        
        if possible:
            partition_choices.append({'partition': partition, 'weight': combinations_for_partition})

    indices = np.arange(len(partition_choices))
    weights_partitions = np.array([p['weight'] for p in partition_choices], dtype=np.int64)
    probabilities_partitions = weights_partitions / np.sum(weights_partitions)
    chosen_index = random_state.choice(indices, p=probabilities_partitions)
    chosen_partition = partition_choices[chosen_index]['partition']
    
    # === 3. 为选定的 Partition 递归构建子树 (后缀列表) ===
    postfix_list = []
    for child_size in chosen_partition:
        child_postfix_list = generate_random_tree(
            child_size, degrees, count_memo, random_state
        )
        postfix_list.extend(child_postfix_list)
    
    postfix_list.append(int(chosen_degree))
    
    return postfix_list


def get_mth_tree(
    size: int,
    degrees: List[int],
    m: int, # 0-based index
    count_memo: Optional[Dict[int, int]] = None
) -> Optional[List[int]]:
    """
    给定确切的大小和索引 m, 返回第 m 个组合的树 (后缀 degree 列表)。
    """
    if count_memo is None:
        count_memo = {}

    total_trees = count_trees(size, degrees, count_memo)
    if m >= total_trees:
        raise IndexError(f"索引 m={m} 超出范围，对于 size={size} 只有 {total_trees} 种可能的树。")

    if size == 1:
        return [0]
    
    nodes_for_children = size - 1
    
    for degree in sorted(degrees): 
        if degree == 0 or nodes_for_children < degree:
            continue
            
        for partition in get_integer_partitions(nodes_for_children, degree):
            block_size = 1
            child_counts = []
            possible = True
            for child_size in partition:
                child_count = count_trees(child_size, degrees, count_memo)
                if child_count == 0:
                    possible = False
                    break
                block_size *= child_count
                child_counts.append(child_count)

            if not possible:
                continue

            if m < block_size:
                postfix_list = []
                remaining_m = m
                for i in range(len(partition)):
                    child_size = partition[i]
                    next_block_divider = block_size // child_counts[i]
                    
                    child_index = remaining_m // next_block_divider
                    child_postfix_list = get_mth_tree(
                        child_size, degrees, child_index, count_memo
                    )
                    postfix_list.extend(child_postfix_list)
                    
                    remaining_m %= next_block_divider
                    block_size = next_block_divider

                postfix_list.append(int(degree))
                return postfix_list
            
            m -= block_size
            
    return None


