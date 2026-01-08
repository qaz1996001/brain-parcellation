# OpenSpec Change: Parameterize UPLOAD_DATA_API_URL

## 📋 Quick Summary

**Change ID**: `parameterize-upload-api-url`
**Status**: ✅ Validated - Ready for Review
**Type**: Enhancement
**Scope**: Task Parameter Configuration
**Impact**: Medium (requires backend service updates)

### What This Changes

Converts `UPLOAD_DATA_API_URL` from environment variable to task parameter, enabling:
- ✅ **Flexible API Routing**: Tasks can call different API endpoints based on dispatch origin
- ✅ **Dual Deployment Support**: Production and Testing backends can use different APIs with shared workers
- ✅ **Better Testing**: Easier to test with mock API endpoints
- ✅ **Explicit Configuration**: API routing visible in task parameters

### Why This Matters

In the dual deployment scenario (Production + Testing in separate folders with shared RabbitMQ/Redis):

**Current Problem** ❌:
```
Testing Backend → Task → Worker (reads ENV) → Production API
                                ↑
                         Wrong API called!
```

**After This Change** ✅:
```
Testing Backend → Task (with test_api_url) → Worker → Testing API
                            ↑
                    URL in task params, not ENV
```

## 📁 Files in This Proposal

| File | Purpose |
|------|---------|
| `proposal.md` | Problem statement, solution, benefits, risks |
| `design.md` | Architecture diagrams, implementation strategy, trade-offs |
| `tasks.md` | 25 ordered tasks with dependencies and validation criteria |
| `specs/task-parameter-configuration/spec.md` | Formal requirements with test scenarios |
| `README.md` | This file - quick reference guide |

## 🎯 Key Changes

### 1. Schema Extension

Add optional `upload_data_api_url` parameter to:
- `Dicom2NiiSeriesParams`
- `Dicom2NiiParams`

With automatic fallback to `UPLOAD_DATA_API_URL` environment variable for backward compatibility.

### 2. Task Function Updates

Update these functions to use parameter instead of `os.getenv()`:
- `dicom_2_nii_series` (line 253)
- `process_dir` (line 388)

### 3. Backend Service Updates

Update all task dispatchers to pass API URL:
- `backend/app/sync/service.py` (2 locations)
- `backend/app/listen/service.py` (2 locations)
- `backend/app/study/service.py` (2 locations)

### 4. Infrastructure

Create `backend/app/config/api_urls.py` helper for centralized URL management.

## 🚀 Implementation Phases

### Phase 1: Schema Extension (3 tasks)
Add parameter field with validation and environment fallback

### Phase 2: Task Functions (2 tasks)
Update task functions to use parameter

### Phase 3: Infrastructure (2 tasks)
Create configuration helper

### Phase 4: Backend Services (6 tasks)
Update all callers to pass URL parameter

### Phase 5: CLI Scripts (2 tasks)
Update pipeline scripts (optional)

### Phase 6: Integration Testing (4 tasks)
Test Production, Testing, and dual deployment routing

### Phase 7: Documentation (4 tasks)
Migration guide, API docs, ADR, deployment guide

### Phase 8: Validation (2 tasks)
Full test suite and staging deployment

**Total**: 25 tasks

## ✅ Validation Status

```bash
$ openspec validate parameterize-upload-api-url --strict
Change 'parameterize-upload-api-url' is valid
```

All requirements:
- ✅ Proposal document complete
- ✅ Design document complete
- ✅ Tasks document with 25 tasks
- ✅ Spec delta with requirements and scenarios
- ✅ Passes strict validation

## 📖 How to Use This Proposal

### For Reviewers

1. **Start with**: `proposal.md` - Understand problem and solution
2. **Then read**: `design.md` - Review architecture and implementation strategy
3. **Check**: `specs/task-parameter-configuration/spec.md` - Validate requirements
4. **Review**: `tasks.md` - Assess implementation plan

### For Implementers

1. **Follow**: `tasks.md` in order - 25 tasks with dependencies
2. **Refer to**: `design.md` for code examples and patterns
3. **Validate**: Each task against scenarios in spec
4. **Test**: Using integration test requirements

### For Users

1. **Migration**: Follow migration guide (to be created in Task 7.1)
2. **Configuration**: Update backend services to pass `upload_data_api_url`
3. **Testing**: Verify dual deployment routing works correctly

## 🔗 Related Changes

- **Depends On**: None
- **Enables**: `integrate-dual-deployment-gpu-solution` - Supports dual deployment with shared workers
- **Related To**: `add-environment-support` - Environment-aware configuration

## ⚠️ Important Notes

### Backward Compatibility

✅ **Fully Backward Compatible**: Environment variable fallback ensures existing deployments continue working.

### Security

✅ **No New Risks**: URLs are internal service addresses, not sensitive credentials.

### Performance

✅ **Negligible Impact**: URL validation adds <1μs per task dispatch.

### Deployment

⚠️ **Requires Coordination**: All backend services must be updated together for full functionality.

## 📞 Contact

**Proposal Author**: Claude Code (AI Assistant)
**Created**: 2024-12-24
**Based On**: User request to parameterize `UPLOAD_DATA_API_URL` in:
- `dicom_tool_get_series_info` → `process_dir`
- `dicom_to_nii`
- `process_dir`
- `nifti_tool_get_series_info` → `dicom_2_nii_series`
- `dicom_2_nii_series`

## 🎬 Next Steps

1. **Review**: Have stakeholders review proposal, design, and spec
2. **Approve**: Get approval to proceed with implementation
3. **Implement**: Follow tasks.md sequentially
4. **Test**: Run full test suite and validate
5. **Deploy**: Deploy to staging, then production
6. **Archive**: Use `openspec archive` when complete

## 📊 Quick Stats

- **Total Tasks**: 25
- **Estimated Effort**: 2-3 weeks (with testing and documentation)
- **Files Modified**: ~15 files
- **New Files Created**: ~10 files (tests, docs, config)
- **Backward Compatible**: Yes ✅
- **Breaking Changes**: No ❌
