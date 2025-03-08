import time
from collections import defaultdict
import functools
import statistics  # Use statistics module instead of numpy

class PerformanceLogger:
    """
    A utility class for tracking execution times of different parts of the code.
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.active_timers = {}
        
    def start_timer(self, name):
        """Start a timer with the given name"""
        self.active_timers[name] = time.perf_counter()
        
    def stop_timer(self, name):
        """Stop a timer with the given name and record the elapsed time"""
        if name in self.active_timers:
            elapsed = time.perf_counter() - self.active_timers[name]
            self.timings[name].append(elapsed)
            del self.active_timers[name]
            return elapsed
        return None
    
    def get_summary(self):
        """Get a summary of all timings"""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'total': sum(times),
                'count': len(times),
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0
            }
        return summary
    
    def print_summary(self):
        """Print a summary of all timings"""
        summary = self.get_summary()
        
        # Sort by total time (descending)
        sorted_items = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print("\n===== PERFORMANCE SUMMARY =====")
        print(f"{'Operation':<30} {'Total (s)':<10} {'Count':<8} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Std (ms)':<10}")
        print("-" * 90)
        
        for name, stats in sorted_items:
            print(f"{name:<30} {stats['total']:<10.3f} {stats['count']:<8} {stats['avg']*1000:<10.2f} {stats['min']*1000:<10.2f} {stats['max']*1000:<10.2f} {stats['std']*1000:<10.2f}")
        
        print("=" * 90)

# Global instance for easy access
performance_logger = PerformanceLogger()

def timed(func=None, name=None):
    """
    Decorator to time a function execution.
    
    Args:
        func: The function to time
        name: Optional name for the timer. If not provided, uses the function name.
    
    Returns:
        The wrapped function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__qualname__}"
            performance_logger.start_timer(timer_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_logger.stop_timer(timer_name)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func) 