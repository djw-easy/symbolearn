"""
分布式计算后端的抽象层
支持 joblib (本地多进程)、Ray (分布式) 和 Dask (分布式)
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Any, Optional


class DistributedBackend(ABC):
    """分布式计算后端的抽象基类"""
    
    @abstractmethod
    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """并行执行函数映射"""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs):
        """初始化后端"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """关闭后端"""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """检查后端是否已初始化"""
        pass

    @abstractmethod
    def scatter(self, data: Any) -> Any:
        """将数据预先分发到工作节点"""
        pass


class JoblibBackend(DistributedBackend):
    """基于 Joblib 的本地多进程后端（默认）"""
    
    def __init__(self, n_jobs: int = 1, verbose: int = 0):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._initialized = False
    
    def initialize(self):
        """Joblib 不需要显式初始化"""
        self._initialized = True
    
    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """使用 Joblib 的 Parallel 执行并行计算"""
        from joblib import Parallel, delayed
        
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(func)(*item) for item in iterable
        )
        return results
    
    def shutdown(self):
        """Joblib 不需要显式关闭"""
        self._initialized = False
    
    def is_initialized(self) -> bool:
        return self._initialized

    def scatter(self, data: Any) -> Any:
        """Joblib 在共享内存中运行，不需要真正的 scatter"""
        return data



class DaskBackend(DistributedBackend):
    """基于 Dask 的分布式后端"""
    
    def __init__(self, cluster, verbose: int = 0):
        self.verbose = verbose
        self._client = None
        self._cluster = cluster
        self._initialized = False

    def initialize(self):
        """初始化 Dask"""
        try:
            from dask.distributed import Client
        except ImportError:
            raise ImportError(
                "Dask is not installed. Please install it with: pip install dask distributed"
            )
        
        # 创建客户端
        self._client = Client(self._cluster)

        if self.verbose > 0:
            print(f"Dask client ready: {self._client}")
        
        self._initialized = True
    
    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """使用 Dask 执行分布式计算"""
        if not self._initialized or self._client is None:
            raise RuntimeError("Dask backend not initialized. Call initialize() first.")
        
        # 提交所有任务
        # Dask's client.map expects arguments to be provided as separate lists,
        # so we need to transpose the iterable of argument tuples.
        # For example, [(a1, b1), (a2, b2)] -> [(a1, a2), (b1, b2)]
        if not iterable:
            return []
        transposed_args = zip(*iterable)
        futures = self._client.map(func, *transposed_args)
        
        # 等待所有任务完成
        results = self._client.gather(futures)
        
        return results
    
    def shutdown(self):
        """关闭 Dask 客户端和集群"""
        if self._client is not None:
            self._client.close()
            self._client = None
        
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None
            
        self._initialized = False
    
    def is_initialized(self) -> bool:
        return self._initialized and self._client is not None

    def scatter(self, data: Any) -> Any:
        """使用 client.scatter 将数据分发到 Dask workers"""
        if not self.is_initialized() or self._client is None:
            raise RuntimeError("Dask backend not initialized. Call initialize() first.")
        # 使用 broadcast=True 确保每个 worker 都有数据副本
        return self._client.scatter(data, broadcast=True)


