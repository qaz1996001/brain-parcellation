# Tasks: Fix Task Queue Consumption Blocking

## Implementation Tasks

### Phase 1: Configuration Changes (1 hour)

- [ ] **Update BoosterParamsMyAI configuration**
  - File: `code_ai/task/params.py`
  - Enable `is_using_distributed_frequency_control = True`
  - Change `concurrent_mode` from `SOLO` to `THREADING`
  - Adjust `concurrent_num` to `3` (reduced from 5)
  - Validation: Configuration matches proposal specification

- [ ] **Update CLAUDE.md documentation**
  - File: `CLAUDE.md`
  - Document distributed frequency control requirements
  - Update GPU mutual exclusion pattern explanation
  - Add Redis dependency note for frequency control
  - Validation: Documentation accurately reflects new behavior

### Phase 2: Testing (2 hours)

- [ ] **Single worker consumption test**
  - Start one worker instance
  - Push 10 test tasks to `task_pipeline_inference_queue`
  - Verify all tasks consumed within 10 seconds
  - Check Redis for active consumer count (should be 1)
  - Validation: `redis-cli GET "consumer_statistics:task_pipeline_inference_queue:active_num"` returns `1`

- [ ] **Multiple worker consumption test**
  - Start two worker instances simultaneously
  - Push 20 test tasks
  - Verify QPS is globally limited to ~1/second (not 2/second)
  - Check Redis for active consumer count (should be 2)
  - Validation: Total consumption rate ≤ 1.2 tasks/second (with tolerance)

- [ ] **Worker restart resilience test**
  - Start worker, push 5 tasks
  - Kill worker after 2 tasks complete
  - Restart worker immediately
  - Verify remaining 3 tasks are consumed without stalling
  - Validation: No messages stuck in Ready state for >10 seconds

- [ ] **Long-running task test**
  - Push task with 120-second inference duration
  - While running, push 5 additional tasks
  - Verify new tasks are prefetched and queued (not stuck in Ready)
  - Validation: RabbitMQ shows messages moving to "Unacked" state

### Phase 3: Monitoring Setup (1 hour)

- [ ] **Add Redis consumer metrics logging**
  - Update `task_pipeline.py` to log active consumer count on startup
  - Add periodic heartbeat status logging (every 60 seconds)
  - Validation: Log shows `Active consumers: N` message

- [ ] **RabbitMQ queue metrics baseline**
  - Document normal Ready/Unacked/Total message counts
  - Set up alert thresholds (e.g., Ready>10 for >5 minutes)
  - Validation: Metrics baseline documented in deployment guide

### Phase 4: Production Rollout (Staged)

- [ ] **Deploy to testing environment**
  - Update testing worker configuration
  - Monitor for 24 hours
  - Check for any consumption stalls or errors
  - Validation: Zero incidents of stuck messages in 24h period

- [ ] **Deploy to production environment**
  - Update production worker configuration
  - Gradual rollout: one worker at a time
  - Monitor GPU utilization and queue depth
  - Validation: Production system stable for 48 hours

### Phase 5: Documentation and Cleanup (30 minutes)

- [ ] **Update troubleshooting guide**
  - Add section on distributed frequency control
  - Document Redis dependency requirements
  - Add debugging commands for active consumer tracking
  - Validation: Guide includes Redis CLI commands for diagnostics

- [ ] **Create runbook for consumption issues**
  - Steps to check Redis heartbeat status
  - Commands to verify active consumer count
  - Procedure for manual worker restart if needed
  - Validation: Runbook tested with operations team

## Validation Checklist

Before marking complete:

- [ ] All configuration changes committed and reviewed
- [ ] All tests passed (single worker, multiple workers, restart, long-running)
- [ ] Documentation updated (CLAUDE.md, troubleshooting guide, runbook)
- [ ] Monitoring shows stable consumption for 48+ hours
- [ ] No regression in GPU mutual exclusion behavior
- [ ] Redis metrics show correct active consumer tracking

## Rollback Plan

If issues occur:

1. Revert `code_ai/task/params.py` to previous version:
   ```python
   concurrent_mode: str = ConcurrentModeEnum.SOLO
   is_using_distributed_frequency_control: bool = False
   ```

2. Restart all workers to apply reverted configuration

3. Monitor for return to previous behavior (including original consumption blocking issue)

## Success Metrics

- **Consumption Latency**: 95th percentile message-to-execution time <5 seconds
- **Stuck Messages**: Zero messages in Ready state for >60 seconds
- **System Uptime**: No manual intervention required for 7 days
- **GPU Utilization**: Maintains 1 concurrent task maximum (no regression)

## Dependencies

- Redis server must be running and accessible
- RabbitMQ connection stable
- No changes required to task functions or pipeline code
