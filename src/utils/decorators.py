"""
Custom Decorators for utility functions.

This module provides reusable decorators to improve logging, performance monitoring,
error handling, and result caching. It is intended to support robust and efficient
data science, machine learning and generate software development projects.

Decorators included:
    - timer: Measure function exec time
    - memory_monitor: Track memory usage before and after func exec
    - retry: Retry a func with exponential backoff in case of failure
    - validate_input: Validate func inputs with a users-defined val func
    - cache_result: Cache func results to disk for faster reuse.

Dependencies:
    - loguru: for structured logging.
    - psutil: for memory monitoring.
    - gc: garbage collection (manual cleanup).
    - functools, time, hashlib, pickle, pathlib.

Author: Elio Le
Date: 30/09/2025
"""
import functools
import time
import traceback
import psutil
import gc
import hashlib
import pickle
from pathlib import Path
from typing import Callable, Any, TypeVar
from loguru import logger

# Generic return type variable
R = TypeVar("R")

# ------------------------------------------------------------------------------
# 1. Timer decorator
# ------------------------------------------------------------------------------
def timer(func: Callable[..., R]) -> Callable[..., R]:
    """
    Measure the execution time of a function and log the result.

    Args:
        func (Callable): The target function to decorate.

    Returns:
        Callable: A wrapped function that logs execution start, end, duration,
                  and raises any exceptions encountered.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> R:
        start_time = time.perf_counter()
        logger.info(f"Starting {func.__name__}...")

        try:
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            logger.success(f"{func.__name__} completed in {elapsed_time:.2f} seconds")

            # Log performance metrics (could be redirected to a separate log file)
            logger.bind(performance=True).info(f"{func.__name__},{elapsed_time:.2f}")
            return result

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed_time:.2f} seconds: {str(e)}"
            )
            raise

    return wrapper


# ------------------------------------------------------------------------------
# 2. Memory monitor decorator
# ------------------------------------------------------------------------------
def memory_monitor(func: Callable[..., R]) -> Callable[..., R]:
    """
    Monitor the memory usage of a function during execution.

    Logs memory consumption before and after execution, and runs garbage
    collection if memory usage exceeds a threshold (default: 500 MB).

    Args:
        func (Callable): The target function to decorate.

    Returns:
        Callable: A wrapped function with memory monitoring.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> R:
        # Get memory usage before function execution
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)

            # Get memory usage after function execution
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before

            logger.info(f"{func.__name__} memory usage: {mem_used:.2f} MB")

            # Trigger garbage collection if memory usage is unusually high
            if mem_used > 500:  # threshold = 500 MB
                gc.collect()
                logger.warning(
                    f"High memory usage detected ({mem_used:.2f} MB). "
                    "Running garbage collection."
                )

            return result

        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


# ------------------------------------------------------------------------------
# 3. Retry decorator
# ------------------------------------------------------------------------------
def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Retry a function with exponential backoff on failure.

    Args:
        max_attempts (int, optional): Maximum number of attempts. Defaults to 3.
        delay (float, optional): Initial wait time before retrying (in seconds).
                                 Wait time doubles after each failed attempt.

    Returns:
        Callable: A decorator that retries the wrapped function.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise

                    wait_time = delay * (2 ** attempt)  # exponential backoff
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)

            return None
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# 4. Input validation decorator
# ------------------------------------------------------------------------------
def validate_input(validation_func: Callable[..., bool]) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Validate function inputs using a custom validation function.

    Args:
        validation_func (Callable): A function that returns True if inputs are valid,
                                    otherwise False.

    Returns:
        Callable: A decorator that validates inputs before executing the target function.
                  Raises ValueError if validation fails.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            if not validation_func(*args, **kwargs):
                raise ValueError(f"Invalid input for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# 5. Caching decorator
# ------------------------------------------------------------------------------
def cache_result(cache_dir: Path = Path("cache")) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Cache function results to disk using pickle.

    If a cached result exists for the same input arguments, the result
    will be loaded instead of recomputing.

    Args:
        cache_dir (Path, optional): Directory to store cache files. Defaults to "cache".

    Returns:
        Callable: A decorator that enables disk-based caching for the target function.
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            # Generate unique cache key based on function name + args + kwargs
            cache_key = hashlib.md5(
                f"{func.__name__}_{args}_{kwargs}".encode()
            ).hexdigest()

            cache_file = cache_dir / f"{cache_key}.pkl"
            cache_dir.mkdir(exist_ok=True)

            # If cache exists, load and return
            if cache_file.exists():
                logger.info(f"Loading {func.__name__} result from cache")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

            # Otherwise compute result
            result = func(*args, **kwargs)

            # Save result to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            return result
        return wrapper
    return decorator