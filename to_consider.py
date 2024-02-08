# ADD A FUNCTION "get_local dir" - j'oublie à chaque fois comment c'est fait.
# Remove le putain de warning
# Escape properly \n dans remove_break_lines


def log_function_execution_time(execution_times: List=[],log_file: str = "") -> Callable:
    """
    Decorator function to log the execution time and output of a wrapped function.

    Args:
        log_file (str): The name of the log file. If not specified, it will create "results_current_time.py" in a "logs" folder.

    Returns:
        Callable: The decorated function.
    """
    if not log_file:
        now = get_now(True)
        log_directory = "logs"
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)
        log_file = f"{log_directory}/results_{now}.py"
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            try:
                result = func(*args, **kwargs)
                function_output = f"{result}"
            except Exception as e:
                log_issue(e, func, f"{function_name} execution failed")
                result = None
                function_output = "Error occurred"
            end_time = time.time()
            execution_time = round((end_time - start_time), 2)
            execution_times.append(execution_time)
            log_message = f"# {function_name}: Execution Time: {execution_time} sc ** Output: {function_output}\n"   
            with open(log_file, 'a') as file:
                file.write(log_message)
            return result
        return wrapper
    return decorator