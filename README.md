# Player Tracking Project

## Installation

### Step 1: Install Anaconda and Setup condarc as follows:
```shell
cd
nano .condarc
```
```yaml
channels:
  - pytorch
  - conda-forge
  - anaconda
  - defaults
channel_priority: strict
```

### Step 2: Create conda environment
```shell
conda create -yn <env_name> python=3.10
conda activate <env_name>
```

### Step 3: Install pytorch (based on your system using pip)
https://pytorch.org/get-started/locally/

### Step 4: Install Yolo
```pip3 install ultralytics```

### Step 5: Install tensorflow
```pip3 install tensorflow scikit-learn```

### Step 6: Run & Enjoy ☺️
```shell
python3 main-gui.py
```

## Performance Logging

The project includes a performance logging system to help identify bottlenecks in the tracking process. The system measures execution times of different parts of the code and provides a summary report at the end of processing.

### Using Performance Logging

Performance logging is automatically enabled when running the application. At the end of processing, a summary report will be displayed showing:

- Total execution time for each operation
- Number of times each operation was executed
- Average, minimum, maximum, and standard deviation of execution times

### Adding Custom Timing

You can add custom timing to your code using the performance logger:

```python
from utils.performance import performance_logger, timed

# Method 1: Manual timing
performance_logger.start_timer("my_operation")
# ... your code here ...
performance_logger.stop_timer("my_operation")

# Method 2: Using the decorator
@timed
def my_function():
    # ... your code here ...
    pass

# Method 3: Using the decorator with a custom name
@timed(name="custom_operation_name")
def another_function():
    # ... your code here ...
    pass
```

See `utils/example_timed.py` for a complete example of how to use the performance logging system.