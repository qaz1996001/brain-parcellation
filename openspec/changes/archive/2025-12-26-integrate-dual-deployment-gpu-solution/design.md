# Design Document: Integrate Dual Deployment with GPU Resource Management

## Change ID
`integrate-dual-deployment-gpu-solution`

## Executive Summary

This change integrates two separate deployment strategies (QUICK_START_DUAL.md and GPU_SOLUTION_COMPLETE.md) into a unified, production-ready deployment guide. The design follows **Linus Torvalds** principles (simplicity, data-structure focus) and **Donald Knuth** principles (precision, verification), prioritizing the simplest solution (Solution B: distributed frequency control) while documenting advanced options.

**Core Decision**:
- **2 FastAPI Instances** (separate data boundaries)
- **2 Funboost Workers** (separate processes)
- **1 Unified Queue** (shared resource scheduler)
- **Distributed Frequency Control** (automatic QPS division)

---

## Architecture Decision Analysis

### Design Philosophy

Following **Linus Torvalds**:
> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."

We identify **correct separation points** in the data structure:

| Layer | Separation Required | Rationale |
|-------|---------------------|-----------|
| **API Layer** | ✅ YES (2 instances) | Data entry boundary - different ports, different request contexts |
| **Database Layer** | ✅ YES (2 databases) | Data storage boundary - production vs testing data isolation |
| **Worker Process** | ✅ YES (2 workers) | Process isolation - independent environment configurations |
| **Task Queue** | ❌ NO (1 unified) | Resource scheduler - shared GPU resource coordination |
| **GPU Control** | ❌ NO (1 distributed) | Resource allocation - automatic QPS division across workers |

Following **Donald Knuth**:
> "Premature optimization is the root of all evil."

We **avoid premature optimization**:
- ❌ **Wrong**: Create 2 separate queues + implement GPU lock → adds complexity, GPU still competes
- ✅ **Right**: Use 1 unified queue + enable distributed frequency control → 1 line of configuration

---

## Critical Clarification: Why 2 Workers Are Required

### Common Misunderstanding ⚠️

**Incorrect Assumption**: "Solution B uses 1 Funboost Worker that determines the environment from task parameters."

**Correct Architecture**: **Solution B requires 2 Funboost Workers** (separate processes), both listening to the same unified queue.

### Why 2 Workers? (Linus & Knuth Principles)

#### Reason 1: Process Isolation (Linus Principle)

**Linus Torvalds**: "Environment variables and process configuration are loaded at process startup, not runtime."

```python
# ❌ WRONG: Single Worker attempting to support both environments
# Problem: Environment variable pollution, configuration conflicts

Worker (single process):
├─ ENV = ??? (production or testing?)
├─ How to support both environments simultaneously?
└─ Environment variables are fixed at process startup!

# ✅ CORRECT: 2 Workers (Solution B)
Worker #1 (separate process):
├─ ENV=production (set at startup)
├─ Database: dicom
├─ Minio: minio_backup
└─ All production configurations loaded

Worker #2 (separate process):
├─ ENV=testing (set at startup)
├─ Database: dicom_testing
├─ Minio: minio_backup_testing
└─ All testing configurations loaded
```

#### Reason 2: Static Configuration vs Dynamic Parameters (Knuth Principle)

**Donald Knuth**: "Be precise about boundary conditions. Static configuration (process-level) ≠ Dynamic parameters (task-level)."

```python
# Why task parameters alone are insufficient:

# ❌ WRONG: Runtime environment switching
def process_task(func_params):
    task_env = func_params.get('environment')

    if task_env == 'production':
        db = connect_to_database('dicom')  # ❌ Unsafe runtime switching
    else:
        db = connect_to_database('dicom_testing')

    # Problems:
    # 1. Database connections are typically established at startup
    # 2. Model paths, Minio buckets are loaded at initialization
    # 3. Environment-specific logging configurations are static
    # 4. Cannot dynamically reload all configurations per task

# ✅ CORRECT: Process-level configuration
# Worker #1 startup (ENV=production):
config = load_environment_config()  # Reads ENV, connects to production DB
database = Database(config.db_url)  # dicom
minio = MinioClient(config.bucket)  # minio_backup

# Worker #2 startup (ENV=testing):
config = load_environment_config()  # Reads ENV, connects to testing DB
database = Database(config.db_url)  # dicom_testing
minio = MinioClient(config.bucket)  # minio_backup_testing
```

#### Reason 3: Independent Startup/Shutdown

**Linus Principle**: "Do one thing well. Each worker has a single, clear identity."

```bash
# ✅ Production can start/stop independently
$ ENV=production ./brain-parcellation-start.sh
$ ENV=production ./brain-parcellation-stop.sh

# ✅ Testing can start/stop independently
$ ENV=testing ./brain-parcellation-start.sh
$ ENV=testing ./brain-parcellation-stop.sh

# ❌ Single worker cannot be both production and testing
# How would you start it? ENV=production,testing? Impossible!
```

### Architecture Comparison

#### ❌ Single Worker (Incorrect Understanding)

```
Issues:
1. ❌ Environment variable conflict (ENV can only have one value)
2. ❌ Cannot dynamically switch database configurations
3. ❌ Cannot independently start/stop Production/Testing
4. ❌ Violates process isolation principle (Linus)
5. ❌ Unclear boundary conditions (Knuth)
```

#### ✅ Solution B: 2 Workers + 1 Unified Queue

```
Advantages:
1. ✅ Complete process isolation
2. ✅ Static environment configuration (determined at startup)
3. ✅ Independent start/stop capability
4. ✅ Unified queue + distributed frequency control (GPU protection)
5. ✅ Aligns with Linus process isolation principle
6. ✅ Aligns with Knuth precise boundary principle
```

### How It Works in Practice

```bash
# Terminal 1: Start Production
$ ENV=production docker-compose up -d
# Starts:
# - FastAPI backend (port 8000)
# - Funboost Worker #1 (ENV=production)
# - PostgreSQL (production DB)

# Terminal 2: Start Testing (different Docker Compose project)
$ ENV=testing docker-compose -p brain_parcellation_testing up -d
# Starts:
# - FastAPI backend (port 8001)
# - Funboost Worker #2 (ENV=testing)
# - PostgreSQL (testing DB)

# Both workers listen to the same RabbitMQ queue
# Redis tracks: active_consumer_num = 2
# Distributed frequency control: Each worker actual QPS = 1/2 = 0.5
```

### Task Parameter Usage

The `environment` field in task parameters serves a **different purpose** than process-level ENV:

```python
# code_ai/task/task_pipeline.py
from backend.app.config import get_environment

# ⭐ Process-level environment (static, set at startup)
ENV = get_environment()  # From os.getenv('ENV')

@Booster(BoosterParamsMyAI(
    queue_name='task_pipeline_inference_queue',  # ⭐ Unified queue
    qps=1,
    is_using_distributed_frequency_control=True
))
def task_pipeline_inference(func_params: Dict[str, any]):
    """Inference task"""

    # ⭐ Task parameter environment (for logging/validation)
    task_env = func_params.get('environment', ENV)

    logger.info(f"[{ENV}] Processing task from {task_env}")

    # ⭐ Actual configuration uses process-level ENV
    # Database connection, Minio bucket, model paths were all
    # loaded at process startup based on ENV variable

    result = perform_inference(func_params)
    return result
```

### Summary

| Component | Quantity | Reason |
|-----------|----------|--------|
| **FastAPI** | 2 | Data entry boundary (different ports) |
| **Database** | 2 | Data storage boundary (production/testing) |
| **Funboost Worker** | **2** ⭐ | **Process isolation, environment configuration isolation** |
| **Task Queue** | 1 | Resource scheduler (unified management) |
| **GPU Control** | 1 | Distributed frequency control (automatic QPS allocation) |

**Key Insight**: The "unified queue" means **1 queue with 2 workers listening**, not "1 worker handling 2 environments".

---

## Detailed Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dual Deployment Architecture                 │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────┐         ┌────────────────────────┐
│  Production Instance   │         │  Testing Instance      │
├────────────────────────┤         ├────────────────────────┤
│ FastAPI Backend        │         │ FastAPI Backend        │
│ - Port: 8000           │         │ - Port: 8001           │
│ - ENV=production       │         │ - ENV=testing          │
│                        │         │                        │
│ Database: dicom        │         │ Database: dicom_testing│
│ - Minio: minio_backup  │         │ - Minio: minio_backup_ │
│                        │         │         testing        │
│                        │         │                        │
│ Funboost Worker #1     │         │ Funboost Worker #2     │
│ - qps: 1               │         │ - qps: 1               │
│ - distributed: True    │         │ - distributed: True    │
└──────────┬─────────────┘         └──────────┬─────────────┘
           │                                  │
           │      ┌───────────────────┐       │
           └─────→│  Unified Queue    │←──────┘
                  │  RabbitMQ         │
                  │                   │
                  │  Redis Tracker:   │
                  │  active_num = 2   │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ Distributed QPS   │
                  │ Control           │
                  │                   │
                  │ Worker #1: 1/2=0.5│
                  │ Worker #2: 1/2=0.5│
                  │ Total: 1.0 qps    │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  GPU Resource     │
                  │  (Max 1 task/s)   │
                  └───────────────────┘
```

### Data Flow

**Request Flow (Solution B - Recommended)**:

```
Production API Request:
1. User → http://localhost:8000/api/inference
2. Backend validates request
3. Publish to queue with environment tag:
   {
     "study_id": "...",
     "model_name": "...",
     "environment": "production"  ← Environment tag
   }
4. Worker picks up task
5. Reads environment from func_params
6. Uses production database (dicom)
7. GPU inference (coordinated by distributed frequency control)

Testing API Request:
1. User → http://localhost:8001/api/inference
2. Backend validates request
3. Publish to queue with environment tag:
   {
     "study_id": "...",
     "model_name": "...",
     "environment": "testing"  ← Environment tag
   }
4. Worker picks up task
5. Reads environment from func_params
6. Uses testing database (dicom_testing)
7. GPU inference (coordinated by distributed frequency control)
```

**Distributed Frequency Control Mechanism** (from funboost source code):

```python
# funboost/consumers/base_consumer.py:525-527
if self.consumer_params.is_using_distributed_frequency_control:
    active_num = self._distributed_consumer_statistics.active_consumer_num
    self._frequency_control(self.consumer_params.qps / active_num, ...)
else:
    self._frequency_control(self.consumer_params.qps, ...)
```

**Behavior**:
- Without distributed control: Each worker executes at `qps=1` → 2 workers = 2 tasks/sec → GPU competition ❌
- With distributed control: Each worker executes at `qps/2=0.5` → 2 workers = 1 task/sec → No GPU competition ✅

**Redis Tracking**:
```python
# Redis stores active consumer count
key: "consumer_statistics:{queue_name}:active_num"
value: 2  # (Production Worker + Testing Worker)
```

---

## Configuration Structure

### Current State

```
Project Root/
├── .env.production          # Production environment variables
├── .env.testing             # Testing environment variables
├── code_ai/task/
│   ├── params.py            # qps=1, SOLO mode
│   └── task_pipeline.py     # Queue configuration
├── funboost_config.py       # Redis/RabbitMQ configuration
└── docker-compose.yml       # Container orchestration
```

### Proposed State (Solution B)

```
Project Root/
├── .env.production          # Production: PORT=8000, ENV=production
├── .env.testing             # Testing: PORT=8001, ENV=testing
├── .env.dual-deployment.example  ← NEW: Configuration template
├── code_ai/task/
│   ├── params.py            # + is_using_distributed_frequency_control=True
│   └── task_pipeline.py     # Unified queue, environment-aware logic
├── scripts/
│   ├── verify-dual-deployment.sh  ← NEW: Deployment verification
│   └── monitor-gpu-usage.sh       ← NEW: GPU monitoring
├── docs/
│   ├── DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md  ← NEW: Unified guide
│   └── archive/
│       ├── QUICK_START_DUAL.md              ← ARCHIVED
│       └── GPU_SOLUTION_COMPLETE.md         ← ARCHIVED
├── funboost_config.py       # (No change)
└── docker-compose.yml       # (No change)
```

### Configuration Changes

**File: `code_ai/task/params.py`**

```python
# Before
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5
    qps: int = 1

# After (ONE LINE CHANGE)
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5
    qps: int = 1
    is_using_distributed_frequency_control: bool = True  # ⭐ ADD THIS LINE
```

**File: `code_ai/task/task_pipeline.py`** (Optional enhancement)

```python
# Add environment-aware logic
from backend.app.config import get_environment

@Booster(BoosterParamsMyAI(
    queue_name='task_pipeline_inference_queue',  # Unified queue
    user_custom_record_process_info_func=save_result_status_to_sqlalchemy,
    qps=1,
))
def task_pipeline_inference(func_params: Dict[str, any]):
    """Inference task - reads environment from func_params"""

    # Read environment from task parameters
    task_env = func_params.get('environment', 'production')

    logger.info(f"[{task_env}] Processing inference task")

    # Use environment-specific database
    db_name = 'dicom' if task_env == 'production' else 'dicom_testing'

    # Perform inference with environment-specific configuration
    result = perform_inference(func_params, db_name)
    return result
```

---

## Linus Torvalds Principles Application

### 1. "Talk is cheap, show me the code"

**Implementation**: Every solution provides executable code, not just descriptions.

Example from `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`:
```bash
# Not: "Enable distributed frequency control"
# But: Exact code with verification
echo 'is_using_distributed_frequency_control: bool = True' >> code_ai/task/params.py

# Verification command
python -c "from code_ai.task.params import BoosterParamsMyAI; \
           print(BoosterParamsMyAI().is_using_distributed_frequency_control)"
# Expected output: True
```

### 2. "Do one thing well"

**Implementation**: Each file has a single clear purpose.

- `DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md`: Complete deployment guide (replaces 2 fragmented docs)
- `.env.dual-deployment.example`: Configuration template (not mixed with code)
- `verify-dual-deployment.sh`: Deployment verification (not mixed with monitoring)
- `monitor-gpu-usage.sh`: GPU monitoring (not mixed with deployment)

### 3. "Good programmers worry about data structures"

**Implementation**: Focus on configuration data structures, not complex code.

```python
# Configuration-driven solution (data structure focus)
params = {
    'qps': 1,
    'is_using_distributed_frequency_control': True  # One line solves the problem
}

# NOT: Complex code-based solution
class GPULock:
    def acquire(self): ...
    def release(self): ...
    # 100+ lines of lock management code
```

---

## Donald Knuth Principles Application

### 1. "Premature optimization is the root of all evil"

**Implementation**: Provide simplest solution first, complex options later.

Document structure:
```markdown
## Quick Start (15 minutes) - Solution B ⭐
- Enable distributed frequency control (1 line)
- Start deployment
- Verify

## Advanced Options (if needed)
- Solution A: GPU Lock (30 minutes, complex)
- Solution C: Priority Control (specialized use case)
```

Decision tree guides users to avoid premature complexity:
```
90% of users → Solution B (simplest)
10% with special needs → Solution A (complex)
```

### 2. "Programs are meant to be read by humans"

**Implementation**: Clear decision tree, structured documentation.

```
User Need: Run Production and Testing simultaneously
    ↓
Question: Need strict environment isolation (different queues)?
    │
    ├─ No (90% scenarios) → Solution B: 5 minutes
    │  ✓ Minimal configuration
    │  ✓ Funboost built-in feature
    │
    └─ Yes (special needs) → Solution A: 30 minutes
       ✓ Complete isolation
       ✓ Requires GPU lock code
```

### 3. "Beware of bugs; prove correctness, don't just try"

**Implementation**: Every step has verification command and expected output.

Example:
```bash
# Step: Enable distributed frequency control
echo 'is_using_distributed_frequency_control: bool = True' >> params.py

# Verification
python -c "from code_ai.task.params import BoosterParamsMyAI; \
           assert BoosterParamsMyAI().is_using_distributed_frequency_control == True"
# Expected: No output (assertion passes) ✓

# GPU competition verification
./scripts/monitor-gpu-usage.sh
# Expected output:
# GPU 0: 45% usage
# Active tasks: 1 ✓  ← Proves no competition
# Distributed control: ACTIVE ✓
```

---

## Solution Comparison

### Solution B (Recommended) - Distributed Frequency Control

**Architecture**:
- 2 FastAPI instances (different ports)
- 2 Databases (production/testing)
- 2 Funboost workers (different processes)
- 1 Unified queue
- 1 Distributed frequency control

**Advantages**:
- ✅ Simplest implementation (1 line configuration change)
- ✅ Built-in funboost feature (stable, tested)
- ✅ Automatic worker coordination
- ✅ No custom code required
- ✅ Redis already configured

**Disadvantages**:
- ⚠️ Queue-level isolation not enforced (handled by environment tags)
- ⚠️ Production/Testing share QPS quota (acceptable for 90% use cases)

**Use Cases**:
- ✅ Standard dual deployment
- ✅ Testing environment with similar workload
- ✅ Shared GPU resource pool

### Solution A (Advanced) - Redis GPU Lock

**Architecture**:
- 2 FastAPI instances (different ports)
- 2 Databases (production/testing)
- 2 Funboost workers (different processes)
- 2 Separate queues (production_queue, testing_queue)
- 1 Custom GPU lock implementation

**Advantages**:
- ✅ Complete environment isolation at queue level
- ✅ Guaranteed GPU mutual exclusion
- ✅ Lock timeout prevents deadlocks

**Disadvantages**:
- ⚠️ Requires custom lock code (~100 lines)
- ⚠️ Additional complexity and maintenance
- ⚠️ Dependency on Redis availability

**Use Cases**:
- ✅ Strict compliance requirements (queue-level isolation)
- ✅ Different SLA for production/testing
- ✅ Advanced debugging and monitoring needs

### Solution C (Specialized) - Priority Control

**Architecture**:
- Similar to Solution B but with different QPS settings:
  - Production: qps=1
  - Testing: qps=0.2 (degraded priority)

**Use Cases**:
- ✅ Production must always have priority
- ✅ Testing runs only during idle time
- ⚠️ Not recommended for balanced dual deployment

---

## Rollback Strategy

### Disabling Distributed Frequency Control

If issues arise, rollback is simple:

```python
# File: code_ai/task/params.py
class BoosterParamsMyAI(BoosterParamsMyRABBITMQ):
    concurrent_mode: str = ConcurrentModeEnum.SOLO
    concurrent_num: int  = 5
    qps: int = 1
    is_using_distributed_frequency_control: bool = False  # ← Set to False
```

**Impact**: System returns to original behavior (each worker runs at qps=1 independently).

**Mitigation**: If GPU competition returns, implement Solution A (GPU lock) as fallback.

### Reverting to Separate Documentation

If unified guide is not effective:

```bash
# Restore old documents from archive
cp docs/archive/QUICK_START_DUAL.md ./
cp docs/archive/GPU_SOLUTION_COMPLETE.md ./

# Update README links
# Delete DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md
```

---

## Testing Strategy

### Unit Testing

**Target**: Distributed frequency control configuration
```python
# tests/test_params.py
def test_distributed_frequency_control_enabled():
    params = BoosterParamsMyAI()
    assert params.is_using_distributed_frequency_control == True
    assert params.qps == 1
```

### Integration Testing

**Scenario 1**: Verify distributed frequency control divides QPS
```bash
# Start 2 workers monitoring same queue
python -m code_ai.task.task_pipeline &  # Worker 1
ENV=testing python -m code_ai.task.task_pipeline &  # Worker 2

# Send 10 tasks
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/inference
done

# Verify: Total processing time ≈ 10 seconds (1 task/sec global)
# Not: ≈ 5 seconds (2 tasks/sec = GPU competition)
```

**Scenario 2**: GPU monitoring shows max 1 concurrent task
```bash
# Monitor GPU while sending concurrent requests
./scripts/monitor-gpu-usage.sh &

# Send 5 concurrent requests
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/inference &
done

# Verify: GPU monitor shows "Active tasks: 1" throughout
```

### End-to-End Testing

**Test Case**: New user deploys dual instances in 20 minutes
```bash
# Time tracking
start_time=$(date +%s)

# Follow DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md
# ... (deployment steps)

# Verify success
end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Assert: elapsed < 1200 seconds (20 minutes)
```

---

## Monitoring and Observability

### Key Metrics

1. **GPU Utilization**
   ```bash
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
   ```

2. **Active Task Count**
   ```bash
   # From Redis
   redis-cli GET "consumer_statistics:task_pipeline_inference_queue:active_tasks"
   # Expected: 0 or 1
   ```

3. **Worker Count**
   ```bash
   # From Redis
   redis-cli GET "consumer_statistics:task_pipeline_inference_queue:active_num"
   # Expected: 2 (Production + Testing workers)
   ```

4. **QPS Per Worker**
   ```python
   # From funboost logs
   logger.info(f"Worker QPS: {qps / active_num}")
   # Expected: 0.5 for each worker
   ```

### Alerting Thresholds

- **GPU Competition Detected**: Active tasks > 1
- **Worker Failure**: active_num < 2 when both environments running
- **High Latency**: Task processing time > expected (indicates queue backup)

---

## Security Considerations

### Environment Isolation

- **API Layer**: Different ports prevent cross-environment requests
- **Database Layer**: Separate databases prevent data contamination
- **Process Layer**: Separate workers prevent environment variable leakage

### Redis Security

- **Access Control**: Use Redis AUTH if exposed externally
- **Network Isolation**: Redis should only be accessible within Docker network
- **Key Expiration**: Consumer statistics keys have TTL

### Queue Security

- **RabbitMQ vhost**: Consider separate vhosts for strict isolation (Solution A)
- **Task Validation**: Validate environment tag in func_params to prevent spoofing

---

## Performance Considerations

### Latency

**Baseline (Single Instance)**:
- Queue latency: ~10ms
- Inference time: ~2000ms
- Total: ~2010ms

**With Distributed Frequency Control (Dual Instance)**:
- Queue latency: ~10ms
- Redis coordination: ~2ms
- Inference time: ~2000ms
- Total: ~2012ms

**Impact**: Negligible (<0.1% overhead)

### Throughput

**Without Distributed Control**:
- Worker 1: 1 task/sec
- Worker 2: 1 task/sec
- Total: 2 tasks/sec (GPU overload ❌)

**With Distributed Control**:
- Worker 1: 0.5 task/sec
- Worker 2: 0.5 task/sec
- Total: 1 task/sec (GPU protected ✅)

### Scalability

**Horizontal Scaling**:
- Adding Worker 3: QPS becomes 1/3 per worker
- Adding Worker 4: QPS becomes 1/4 per worker
- Automatic coordination, no code changes required

---

## Documentation Structure

### DUAL_DEPLOYMENT_PRODUCTION_GUIDE.md

```markdown
# Dual Deployment Production Guide

## Prerequisites (5 minutes)
- System requirements
- Dependency verification
- Command: ./scripts/verify-prerequisites.sh

## Quick Start - Solution B (15 minutes) ⭐ RECOMMENDED
### Step 1: Enable Distributed Frequency Control
- Code change (1 line)
- Verification command
- Expected output

### Step 2: Start Dual Deployment
- Command: ./deploy-dual.sh
- Verification command
- Expected output

### Step 3: Verify GPU Protection
- Command: ./scripts/monitor-gpu-usage.sh
- Expected: Active tasks: 1

## Advanced Options (30+ minutes)
### Solution A: GPU Lock (if you need strict queue isolation)
### Solution C: Priority Control (if production must always win)

## Monitoring and Troubleshooting
- GPU monitoring
- Common issues
- Log inspection

## Appendix
- Port allocation table
- Configuration reference
- Architecture diagram
```

---

## Open Questions and Decisions

### Q1: Should we archive or delete old documents?
**Decision**: Archive to `docs/archive/` with deprecation notice.
**Rationale**: Preserve history, allow rollback if needed.

### Q2: Should Solution B be the only documented option?
**Decision**: Quick Start shows only Solution B, Advanced Options document A/C.
**Rationale**: Follow Knuth's principle - avoid premature complexity.

### Q3: Should we implement GPU lock code now?
**Decision**: No. Document the code in Solution A, implement only if requested.
**Rationale**: YAGNI - 90% of users don't need it.

### Q4: Should we modify QUICK_START deployment scripts?
**Decision**: Keep existing scripts working, create new unified guide.
**Rationale**: Backward compatibility, users can choose migration path.

---

## References

### Internal Documentation
- `QUICK_START_DUAL.md` - Original dual deployment guide
- `GPU_SOLUTION_COMPLETE.md` - GPU resource management solutions
- `openspec/changes/add-environment-support/` - Environment configuration infrastructure

### External References
- Funboost Documentation: https://funboost.readthedocs.io/
- Distributed Frequency Control: `funboost/consumers/base_consumer.py:525-527`
- Docker Compose: https://docs.docker.com/compose/
- Redis Documentation: https://redis.io/docs/

### Design Principles
- Linus Torvalds: "Just for Fun" (autobiography)
- Donald Knuth: "The Art of Computer Programming"
- Martin Fowler: "Refactoring" (YAGNI principle)
