# Lightweight promotion output comparison

This is an evidence record, not a human approval gate.

- Automated blocking findings: []
- Exact output-match rate (diagnostic only): 1.000

## python_code_extended

### Prompt

- user: Write a complete Python function named retry_delays that returns exponential retry delays capped at a maximum. Include type hints, a docstring, input validation, and a small executable example call.

### Active output

<pre>
```python
from typing import Callable, Iterable, List, Optional, Union

def retry_delays(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    max_retries: int = 5,
    jitter: bool = False,
    jitter_range: Optional[tuple] = None
) -&gt; Callable[[Callable], Callable]:
    &quot;&quot;&quot;
    A decorator factory that generates exponential backoff retry delays.
    
    This function returns a decorator that can be applied to any function.
    When the decorated function is called, it will retry the function call
    with exponentially increasing delays between attempts, up to a maximum delay.
    
    Parameters
    ----------
    base_delay : float
        The initial delay in seconds before the first retry.
    max_delay : float
        The maximum delay in seconds between retries.
    multiplier : float
        The factor by which the delay increases with each retry.
    max_retries : int
        The maximum number of retry attempts (including the initial call).
    jitter : bool
        If True, adds random jitter to the delay to prevent thundering herd problems.
    jitter_range : tuple, optional
        A tuple (min_jitter, max_jitter) for the random jitter in seconds.
        If not provided and jitter is True, defaults to (0, base_delay * 0.1).
    
    Returns
    -------
    Callable
        A decorator function that wraps the target function and implements retry logic.
    
    Raises
    ------
    TypeError
        If base_delay, max_delay, or multiplier are not positive numbers.
        If max_retries is not a positive integer.
        If jitter_range is provided but not a tuple of two non-negative numbers.
    
    Examples
    --------
    &gt;&gt;&gt; import time
    &gt;&gt;&gt; import random
    
    &gt;&gt;&gt; @retry_delays(base_delay=1, max_delay=10, max_retries=3)
    ... def flaky_function():
    ...     if random.random() &lt; 0.7:
    ...         raise ConnectionError(&quot;Simulated failure&quot;)
    ...     return &quot;Success&quot;
    
    &gt;&gt;&gt; # This would retry up to 3 times with delays: 1s, 2s, 4s (capped at 10s)

</pre>

### Candidate output

<pre>
```python
from typing import Callable, Iterable, List, Optional, Union

def retry_delays(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    max_retries: int = 5,
    jitter: bool = False,
    jitter_range: Optional[tuple] = None
) -&gt; Callable[[Callable], Callable]:
    &quot;&quot;&quot;
    A decorator factory that generates exponential backoff retry delays.
    
    This function returns a decorator that can be applied to any function.
    When the decorated function is called, it will retry the function call
    with exponentially increasing delays between attempts, up to a maximum delay.
    
    Parameters
    ----------
    base_delay : float
        The initial delay in seconds before the first retry.
    max_delay : float
        The maximum delay in seconds between retries.
    multiplier : float
        The factor by which the delay increases with each retry.
    max_retries : int
        The maximum number of retry attempts (including the initial call).
    jitter : bool
        If True, adds random jitter to the delay to prevent thundering herd problems.
    jitter_range : tuple, optional
        A tuple (min_jitter, max_jitter) for the random jitter in seconds.
        If not provided and jitter is True, defaults to (0, base_delay * 0.1).
    
    Returns
    -------
    Callable
        A decorator function that wraps the target function and implements retry logic.
    
    Raises
    ------
    TypeError
        If base_delay, max_delay, or multiplier are not positive numbers.
        If max_retries is not a positive integer.
        If jitter_range is provided but not a tuple of two non-negative numbers.
    
    Examples
    --------
    &gt;&gt;&gt; import time
    &gt;&gt;&gt; import random
    
    &gt;&gt;&gt; @retry_delays(base_delay=1, max_delay=10, max_retries=3)
    ... def flaky_function():
    ...     if random.random() &lt; 0.7:
    ...         raise ConnectionError(&quot;Simulated failure&quot;)
    ...     return &quot;Success&quot;
    
    &gt;&gt;&gt; # This would retry up to 3 times with delays: 1s, 2s, 4s (capped at 10s)

</pre>

### Automated observations

<pre>
{
  &quot;case_id&quot;: &quot;python_code_extended&quot;,
  &quot;baseline_characters&quot;: 2059,
  &quot;candidate_characters&quot;: 2059,
  &quot;output_exact_match&quot;: true,
  &quot;blocking&quot;: [],
  &quot;attention&quot;: []
}
</pre>

## javascript_debug_extended

### Prompt

- user: Fix this JavaScript so it counts only finite numbers, then explain the bug accurately and briefly. Include a runnable corrected snippet and its expected output:
const values = [1, NaN, Infinity, 2];
const count = values.filter(Boolean).length;

### Active output

<pre>
### The Bug
The original code uses `filter(Boolean)`, which treats `Infinity` as a truthy value (because `Boolean(Infinity)` is `true`). Consequently, it counts `Infinity` along with the finite numbers, resulting in a count of 3 instead of 2. Additionally, `NaN` is correctly filtered out because `Boolean(NaN)` is `false`.

To fix this, you must explicitly check if a number is finite using `Number.isFinite()`.

### Corrected Snippet &amp; Output

```javascript
const values = [1, NaN, Infinity, 2];

// Filter only finite numbers
const finiteValues = values.filter(Number.isFinite);
const count = finiteValues.length;

console.log(count); // Output: 2
```
</pre>

### Candidate output

<pre>
### The Bug
The original code uses `filter(Boolean)`, which treats `Infinity` as a truthy value (because `Boolean(Infinity)` is `true`). Consequently, it counts `Infinity` along with the finite numbers, resulting in a count of 3 instead of 2. Additionally, `NaN` is correctly filtered out because `Boolean(NaN)` is `false`.

To fix this, you must explicitly check if a number is finite using `Number.isFinite()`.

### Corrected Snippet &amp; Output

```javascript
const values = [1, NaN, Infinity, 2];

// Filter only finite numbers
const finiteValues = values.filter(Number.isFinite);
const count = finiteValues.length;

console.log(count); // Output: 2
```
</pre>

### Automated observations

<pre>
{
  &quot;case_id&quot;: &quot;javascript_debug_extended&quot;,
  &quot;baseline_characters&quot;: 654,
  &quot;candidate_characters&quot;: 654,
  &quot;output_exact_match&quot;: true,
  &quot;blocking&quot;: [],
  &quot;attention&quot;: []
}
</pre>

