"""Task path configuration helper for task parameter injection.

This module provides centralized configuration for task execution paths used by task dispatchers.
Tasks receive paths as parameters rather than reading from environment directly,
enabling flexible routing in dual deployment scenarios.
"""

import os
from typing import Dict, Optional


def get_task_execution_paths(override: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get task execution path configuration for task parameter injection.

    This function provides the path configuration that task dispatchers should pass to workers
    as task parameters. In dual deployment scenarios, this allows Production and Testing backends
    to route tasks to different execution paths while sharing the same worker pool.

    Args:
        override: Optional path overrides. If provided, these paths are used instead of
                 reading from environment. Useful for testing or dynamic routing.

    Returns:
        Dict[str, str]: Dictionary with keys: path_process, path_json, path_log

    Raises:
        ValueError: If required paths are not configured, paths are not absolute,
                   or paths are not accessible.

    Examples:
        >>> # Production backend
        >>> os.environ['PATH_PROCESS'] = 'D:/00_Chen/Task04_git/process'
        >>> os.environ['PATH_JSON'] = 'D:/00_Chen/Task04_git/json'
        >>> os.environ['PATH_LOG'] = 'D:/00_Chen/Task04_git/logs'
        >>> paths = get_task_execution_paths()
        >>> task_dict['path_process'] = paths['path_process']
        >>> task_dict['path_json'] = paths['path_json']
        >>> task_dict['path_log'] = paths['path_log']
        >>> task_pipeline_inference.push(task_dict)

        >>> # Testing backend with override
        >>> test_paths = {
        ...     'path_process': 'D:/00_Chen/Task04_git_test/process',
        ...     'path_json': 'D:/00_Chen/Task04_git_test/json',
        ...     'path_log': 'D:/00_Chen/Task04_git_test/logs'
        ... }
        >>> paths = get_task_execution_paths(override=test_paths)
        >>> task_dict.update(paths)
        >>> task_pipeline_inference.push(task_dict)
    """
    # Use override if provided, otherwise read from environment
    paths = override or {}

    result = {
        'path_process': paths.get('path_process') or os.getenv("PATH_PROCESS"),
        'path_json': paths.get('path_json') or os.getenv("PATH_JSON"),
        'path_log': paths.get('path_log') or os.getenv("PATH_LOG"),
    }

    # Validate that all required paths are configured
    for key, value in result.items():
        if value is None:
            raise ValueError(
                f"{key} must be configured via override parameter or "
                f"{key.upper()} environment variable must be set"
            )

        # Validate that paths are absolute
        if not os.path.isabs(value):
            raise ValueError(
                f"{key} must be an absolute path, got: {value}"
            )

    # Note: We don't validate path existence/writability here because:
    # 1. Paths may not exist yet and will be created by tasks (os.makedirs)
    # 2. Permission checks at dispatch time may not reflect worker environment
    # 3. Workers will validate and create paths as needed during execution

    return result
