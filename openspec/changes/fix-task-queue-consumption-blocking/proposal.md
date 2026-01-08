# Proposal: Fix Task Queue Consumption Blocking

## Why

The `task_pipeline_inference_queue` occasionally stops consuming messages, causing GPU inference tasks to accumulate in RabbitMQ's "Ready" state without being processed. This blocking behavior stems from two configuration issues: (1) missing distributed frequency control allows multiple workers to conflict over GPU resources, and (2) `SOLO` concurrent mode blocks message prefetch during long-running inference tasks. The fix enables funboost's built-in distributed frequency control and switches to threading mode, ensuring reliable message consumption while maintaining GPU mutual exclusion.

## Problem Statement

The `task_pipeline_inference_queue` occasionally stops consuming messages, causing tasks to remain stuck in "Ready" state in RabbitMQ. This results in:
- GPU inference tasks not being processed despite idle GPU resources
- Messages accumulating in the queue without consumption
- System appearing "stuck" until worker restart

## Root Cause Analysis

### Current Configuration Issues

1. **Missing Distributed Frequency Control**
   - File: `code_ai/task/params.py:29`
   - Current: `is_using_distributed_frequency_control: bool = False`
   - Impact: Each worker independently enforces `qps=1`, but without coordination

2. **SOLO Concurrent Mode Limitation**
   - File: `code_ai/task/params.py:21`
   - Current: `concurrent_mode: str = ConcurrentModeEnum.SOLO`
   - Impact: Single-threaded consumption can block on long-running tasks

3. **Low Concurrent Number**
   - File: `code_ai/task/params.py:22`
   - Current: `concurrent_num: int = 5`
   - Impact: Limited queue prefetch capability

### Why This Causes Blocking

According to funboost documentation (funboost_合并教程.md):
> "celery不支持分布式全局控频，celery的rate_limit 基于单work控频，如果把脚本在同一台机器启动好几次，或者在多个容器里面启动消费，那么总的qps会乘倍数增长。"

The current setup has the same issue - without distributed frequency control:
- Multiple workers (or worker restarts) can create consumption conflicts
- `SOLO` mode blocks message acknowledgment during long inference tasks
- RabbitMQ holds messages in "Ready" state waiting for worker capacity

## Proposed Solution

### Option A: Enable Distributed Frequency Control (Recommended)

**Changes Required:**
1. Enable distributed frequency control in `BoosterParamsMyAI`
2. Ensure Redis connection is properly configured (already present)
3. Enable consumer heartbeat to Redis for active consumer tracking

**Benefits:**
- True distributed QPS coordination across all workers
- Automatic load balancing when multiple workers are active
- Prevents consumption stalls from worker conflicts

**Trade-offs:**
- Adds Redis dependency for frequency control (Redis already required for other features)
- Slight overhead for heartbeat tracking

### Option B: Switch to Threading Mode

**Changes Required:**
1. Change `concurrent_mode` from `SOLO` to `THREADING`
2. Increase `concurrent_num` to allow parallel message processing
3. Keep `qps=1` for GPU mutual exclusion at task execution level

**Benefits:**
- Non-blocking message consumption (prefetch works correctly)
- Better queue throughput
- More resilient to individual task failures

**Trade-offs:**
- Need to ensure GPU mutual exclusion happens at subprocess level (already handled by subprocess execution)

### Option C: Hybrid Approach (Maximum Reliability)

Combine both solutions:
1. Enable distributed frequency control
2. Use threading mode with limited concurrency
3. Maintain `qps=1` for global GPU coordination

## Recommended Approach

**Option C (Hybrid)** provides maximum reliability:

```python
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.THREADING  # Changed from SOLO
    concurrent_num: int = 3  # Reduced from 5 (limited prefetch)
    qps: int = 1  # Unchanged

    # Enable distributed frequency control
    is_using_distributed_frequency_control: bool = True
    is_send_consumer_hearbeat_to_redis: bool = True  # Already True in parent
```

### Why This Works

1. **Threading Mode**: Allows RabbitMQ to prefetch messages without blocking
2. **Limited Concurrency**: Prevents excessive resource usage (3 messages prefetched)
3. **Distributed QPS**: Coordinates frequency control across all workers via Redis
4. **Global GPU Lock**: `qps=1` + distributed control = true GPU mutual exclusion

## Implementation Plan

1. Modify `code_ai/task/params.py` to enable distributed frequency control
2. Update configuration documentation in CLAUDE.md
3. Test with multiple worker scenarios
4. Monitor Redis heartbeat and active consumer metrics
5. Validate queue consumption stability over 24+ hours

## Verification Criteria

Success metrics:
- Messages consistently consumed within 5 seconds of arrival
- No "stuck" messages in Ready state for >1 minute
- Active consumer count correctly tracked in Redis
- GPU utilization maintains 1 concurrent task maximum
- System survives worker restart without consumption stalls

## Dependencies

- Redis connection (already configured via `BrokerConnConfig`)
- RabbitMQ (existing infrastructure)
- No new external dependencies required

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Redis failure stops all consumption | Use Redis cluster/sentinel for HA |
| Heartbeat overhead impacts performance | Heartbeat interval is configurable (default efficient) |
| Multiple workers compete for tasks | Distributed frequency control prevents this |
| Existing behavior changes | Phased rollout with monitoring |

## References

- funboost documentation: Section 2.4.13 (分布式控频)
- Current implementation: `code_ai/task/task_pipeline.py:110-114`
- Configuration: `code_ai/task/params.py:20-29`
- CLAUDE.md GPU Mutual Exclusion Pattern
