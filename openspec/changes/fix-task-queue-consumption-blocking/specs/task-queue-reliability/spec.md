# Specification: Task Queue Reliability

## MODIFIED Requirements

### Requirement: Task Queue Consumer Configuration MUST Enable Distributed Frequency Control

**Context:**
The GPU inference task queue (`task_pipeline_inference_queue`) processes AI inference tasks that require exclusive GPU access. Multiple workers may consume from this queue across different deployment environments (production, testing). Without distributed frequency control, each worker independently enforces QPS limits, leading to GPU resource conflicts and consumption blocking.

**Specification:**
The `BoosterParamsMyAI` configuration class SHALL enable distributed frequency control by setting `is_using_distributed_frequency_control = True`. Workers SHALL use `concurrent_mode = THREADING` to enable non-blocking message prefetch. The `concurrent_num` parameter SHALL be set to 3 to limit prefetch count while maintaining responsiveness.

**File:** `code_ai/task/params.py`

**Change:**
```python
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.THREADING  # Changed from SOLO
    concurrent_num: int = 3  # Reduced from 5 for controlled prefetch
    qps: int = 1  # Unchanged - global GPU mutual exclusion

    # Enable distributed frequency control
    is_using_distributed_frequency_control: bool = True  # Changed from False
```

#### Scenario: Single Worker Consumes Messages Without Blocking

**Given:**
- One worker instance running with `task_pipeline_inference_queue`
- Redis connection is active and accessible
- Worker configured with distributed frequency control enabled

**When:**
- Backend pushes 10 inference tasks to the queue
- Each task takes 5 seconds to complete

**Then:**
- Worker MUST consume all 10 tasks within 60 seconds (10 tasks × 1 qps + overhead)
- Redis MUST show `active_num = 1` for the queue
- No messages MUST remain in "Ready" state for more than 5 seconds
- Worker logs MUST show distributed frequency control is active

#### Scenario: Multiple Workers Coordinate Global QPS

**Given:**
- Two worker instances running with `task_pipeline_inference_queue`
- Both workers have `qps=1` configured
- Distributed frequency control enabled on both workers

**When:**
- Backend pushes 20 inference tasks to the queue
- Both workers are actively consuming

**Then:**
- Redis MUST report `active_num = 2` for the queue
- Each worker MUST process approximately 10 tasks
- Total consumption rate MUST NOT exceed 1.2 tasks/second (global)
- Individual worker effective QPS MUST be approximately 0.5 tasks/second
- GPU utilization MUST show maximum 1 concurrent task at any time

#### Scenario: Worker Restart Does Not Cause Consumption Stall

**Given:**
- Worker A is consuming tasks from the queue
- Redis shows `active_num = 1`
- 10 tasks are in the queue

**When:**
- Worker A processes 3 tasks successfully
- Worker A is killed (simulated crash)
- Worker B starts within 10 seconds

**Then:**
- Remaining 7 tasks MUST be consumed by Worker B
- Redis `active_num` MUST update from 1 → 0 → 1 within heartbeat TTL (60 seconds)
- No tasks MUST remain stuck in "Ready" state for more than 60 seconds
- Worker B MUST automatically assume full `qps=1` rate after heartbeat expires

#### Scenario: Long-Running Task Does Not Block Queue Consumption

**Given:**
- Worker configured with `concurrent_mode=THREADING` and `concurrent_num=3`
- Worker is idle

**When:**
- Backend pushes Task A with 120-second execution time
- Backend immediately pushes Tasks B, C, D (10-second execution each)

**Then:**
- Task A MUST start executing immediately
- Tasks B, C, D MUST transition to "Unacked" state (prefetched)
- Worker MUST NOT block message consumption while Task A runs
- Tasks B, C, D MUST start executing according to QPS throttling (1/second)
- RabbitMQ "Ready" message count MUST NOT increase while worker is active

### Requirement: Distributed Frequency Control MUST Coordinate via Redis

**Context:**
Funboost framework uses Redis to track active consumer counts and coordinate distributed QPS limits. This requires proper Redis connection configuration and heartbeat mechanism.

**Specification:**
Workers SHALL send periodic heartbeats to Redis to maintain active consumer count. The heartbeat interval MUST be 30 seconds with a TTL of 60 seconds. Workers SHALL recalculate effective QPS based on active consumer count retrieved from Redis. When Redis is unavailable, workers SHALL fall back to local QPS control without crashing.

**Dependencies:**
- Redis server MUST be accessible via `BrokerConnConfig.REDIS_HOST` and `REDIS_PORT`
- `is_send_consumer_hearbeat_to_redis` MUST be `True` (inherited from `BoosterParamsMyRABBITMQ`)

#### Scenario: Worker Sends Heartbeat to Redis

**Given:**
- Worker starts with distributed frequency control enabled
- Redis server is running and accessible

**When:**
- Worker begins consuming from `task_pipeline_inference_queue`

**Then:**
- Worker MUST register in Redis within 30 seconds of startup
- Redis key `consumer_statistics:task_pipeline_inference_queue:active_num` MUST increment by 1
- Worker MUST send heartbeat every 30 seconds (configurable interval)
- Worker heartbeat key MUST have TTL of 60 seconds

#### Scenario: Worker Heartbeat Expiration Updates Active Count

**Given:**
- Two workers (A and B) are running
- Redis `active_num = 2`

**When:**
- Worker A crashes without graceful shutdown
- 60 seconds elapse (heartbeat TTL expires)

**Then:**
- Redis `active_num` MUST decrement to 1 automatically
- Worker B effective QPS MUST increase from 0.5 to 1.0
- Worker B MUST recalculate distributed QPS within 30 seconds

#### Scenario: Redis Connection Failure Falls Back Gracefully

**Given:**
- Worker is running with distributed frequency control enabled
- Redis connection becomes unavailable (network partition)

**When:**
- Worker attempts to send heartbeat
- Redis connection fails

**Then:**
- Worker MUST log warning about Redis connection failure
- Worker MUST fall back to local QPS control (`qps=1`)
- Worker MUST continue consuming messages (degraded mode)
- Worker MUST NOT crash or stop consuming

### Requirement: Threading Concurrent Mode MUST Allow Non-Blocking Prefetch

**Context:**
`SOLO` concurrent mode processes messages sequentially in a single thread, which blocks RabbitMQ message prefetch during long-running tasks. This causes messages to remain in "Ready" state indefinitely.

**Specification:**
The consumer SHALL use `concurrent_mode=THREADING` to enable non-blocking message prefetch. RabbitMQ MUST be able to deliver up to `concurrent_num` messages to the worker without blocking. Each message SHALL be processed in an independent thread from the thread pool. Long-running tasks SHALL NOT block acknowledgment of subsequent messages.

**File:** `code_ai/task/params.py`

**Change:**
```python
concurrent_mode: str = ConcurrentModeEnum.THREADING  # Changed from SOLO
concurrent_num: int = 3  # Prefetch up to 3 messages
```

#### Scenario: Threading Mode Enables Message Prefetch

**Given:**
- Worker configured with `concurrent_mode=THREADING` and `concurrent_num=3`
- Queue has 10 tasks waiting

**When:**
- Worker starts consuming

**Then:**
- RabbitMQ MUST deliver up to 3 messages to worker immediately
- Messages MUST transition from "Ready" to "Unacked" state
- Worker thread pool MUST have 3 threads available
- Each thread MUST process one message independently

#### Scenario: Threading Mode Prevents Blocking on Long Tasks

**Given:**
- Worker has `concurrent_mode=THREADING`
- Thread 1 is executing a 120-second task

**When:**
- Backend pushes new task to queue
- Worker has available threads (Thread 2 and Thread 3 idle)

**Then:**
- RabbitMQ MUST deliver new message to worker immediately
- New message MUST be assigned to idle thread (Thread 2 or Thread 3)
- Thread 1 MUST continue executing long task
- Message consumption MUST NOT block on Thread 1's task completion

### Requirement: Configuration Changes MUST Be Documented

**Context:**
The distributed frequency control and threading mode changes represent significant behavioral modifications that impact deployment, monitoring, and troubleshooting.

**Specification:**
CLAUDE.md SHALL document the distributed frequency control requirement in the "GPU Mutual Exclusion Pattern" section. The documentation MUST explain why distributed frequency control is necessary. The documentation SHALL provide Redis monitoring commands for checking active consumer counts. The documentation MUST list Redis as a dependency for proper operation.

**File:** `CLAUDE.md`

**Change:** Add documentation section explaining:
- Distributed frequency control coordinates QPS across multiple workers
- Redis dependency required for proper operation
- Threading mode enables non-blocking message consumption
- Redis monitoring commands for checking active consumer counts

#### Scenario: Documentation Explains Distributed Frequency Control Requirements

**Given:**
- Developer reads CLAUDE.md GPU Mutual Exclusion Pattern section

**When:**
- Developer needs to add new GPU-based task

**Then:**
- Documentation MUST explain why distributed frequency control is required
- Documentation MUST provide Redis monitoring commands
- Documentation MUST clarify that Redis is a dependency
- Documentation MUST show example of active consumer tracking

## ADDED Requirements

None. This change modifies existing configuration and adds documentation.

## REMOVED Requirements

None. This change maintains backward compatibility (degrades gracefully if Redis unavailable).

## RENAMED Requirements

None.
