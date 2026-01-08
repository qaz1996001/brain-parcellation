# Proposal: migrate-to-iec62304

## Summary
Replace ISO 20816 (mechanical vibration monitoring) compliance framework with IEC-62304 (medical device software lifecycle), add RFC 2119 requirement keywords, integrate ISO 27001 security standards, and provide optional ISO 9001 quality management templates.

## Why
The current meta-framework was designed around ISO 20816 for mechanical vibration monitoring software. This change pivots the framework to support medical device software development, which requires:

1. **Medical Device Compliance**: IEC-62304 is the international standard for medical device software lifecycle processes
2. **Precise Requirement Language**: RFC 2119 provides standardized keywords (SHALL, MUST, SHOULD, MAY) for unambiguous requirements
3. **Security Framework**: ISO 27001 ensures information security management for sensitive medical data
4. **Quality Management**: ISO 9001 (optional) provides quality management system foundations

### Business Value
- Enables medical device software development teams to use this framework
- Provides rigorous software safety classification (Class A/B/C) assessment
- Establishes comprehensive traceability from requirements through risk controls
- Supports regulatory submissions (FDA, EU MDR, PMDA)

### Technical Rationale
ISO 20816 focuses on vibration measurement and monitoring, which is domain-specific. IEC-62304 provides broader applicability to any medical device software, including:
- Diagnostic software
- Therapeutic software
- Patient monitoring systems
- Medical data analysis tools

## What Changes?

### 1. Regulation Standards Migration

**REMOVE**:
- `/docs/resource/meta-framework/regulations/ISO-20816/` directory (empty but referenced)
- All ISO 20816 references in documentation

**ADD**:
- `/docs/resource/meta-framework/regulations/IEC-62304/` with templates:
  - `compliance-mapping-template.md` - IEC-62304 compliance checklist
  - `software-safety-classification.md` - Class A/B/C assessment guide
  - `software-requirements-spec-template.md` - SRS aligned with IEC-62304
  - `software-design-spec-template.md` - SDD aligned with IEC-62304
  - `software-test-plan-template.md` - STP aligned with IEC-62304
  - `soup-management-template.md` - Software of Unknown Provenance tracking
  - `risk-management-bridge-template.md` - Integration with ISO 14971

**ADD**:
- `/docs/resource/meta-framework/regulations/RFC-2119/`
  - `requirement-keywords-guide.md` - SHALL, MUST, SHOULD, MAY usage
  - `requirement-examples.md` - Properly formatted requirement examples

**ADD**:
- `/docs/resource/meta-framework/regulations/ISO-27001/`
  - `security-requirements-template.md` - Information security controls
  - `compliance-mapping-template.md` - ISO 27001 compliance checklist
  - `data-protection-guide.md` - Patient data protection requirements

**ADD (Optional)**:
- `/docs/resource/meta-framework/regulations/ISO-9001/`
  - `quality-management-template.md` - QMS process documentation
  - `compliance-mapping-template.md` - ISO 9001 compliance checklist

### 2. Documentation Structure Updates

**MODIFY**:
- `00_METAFRAMEWORK_INDEX.md`:
  - Update § 5 (Standards Compliance) to list IEC-62304, RFC 2119, ISO 27001, ISO 9001 (optional)
  - Remove ISO 20816 references
  - Add IEC-62304 software safety classification explanation
  - Add SOUP management concept

**MODIFY**:
- `01_FRAMEWORK_OVERVIEW.md`:
  - Update § 2.3 (Standards Compliance Framework) table
  - Replace ISO 20816 use case (§ 6) with medical device software example
  - Update compliance checklist (§ 5.3) for IEC-62304

**MODIFY**:
- `README.md`:
  - Update § 📐 (Standards Support) section
  - Replace vibration monitoring example with medical software example
  - Update § 🚀 (Use Cases) - replace Scenario 3

**MODIFY**:
- `regulations/00_REGULATIONS_INDEX.md`:
  - Remove § 4 (ISO 20816) completely
  - Add § 4 (IEC-62304)
  - Add § 5 (RFC 2119)
  - Add § 6 (ISO 27001)
  - Add § 7 (ISO 9001 - Optional)
  - Update § 5.1 (Standards Selection Guide)
  - Update § 6 (Evidence Package Structure)

### 3. Requirements Template Updates

**MODIFY**:
- `requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md`:
  - Add RFC 2119 keyword usage guidelines in requirements section
  - Add IEC-62304 software safety class field
  - Add SOUP tracking section
  - Update traceability matrix to include risk controls
  - Replace vibration measurement examples with medical software examples

**MODIFY**:
- `requirements/HOWTO_WRITE_REQUIREMENTS.md`:
  - Add § RFC 2119 Keyword Usage
  - Add § IEC-62304 Requirement Classification
  - Update examples to medical device context

### 4. Architecture and Design Updates

**MODIFY**:
- `architecture/` templates (when created):
  - Add IEC-62304 SDD (Software Design Description) alignment
  - Add SOUP integration design patterns
  - Add security architecture section (ISO 27001)

### 5. Guides Updates

**MODIFY**:
- `guides/AI_COLLABORATION_PATTERNS.md`:
  - Update compliance references from ISO 20816 to IEC-62304
  - Add SOUP evaluation AI workflow
  - Add security requirement review checklist

**MODIFY**:
- `guides/TRACEABILITY_MANAGEMENT.md` (when created):
  - Add risk control traceability (IEC-62304 + ISO 14971)
  - Add SOUP traceability requirements

## Impact Analysis

### Affected Capabilities
1. **meta-framework-compliance** (NEW) - Core compliance framework
2. **requirement-engineering** (MODIFIED) - RFC 2119 keywords added
3. **traceability-management** (MODIFIED) - Risk control tracing added
4. **documentation-templates** (MODIFIED) - All templates updated
5. **ai-collaboration** (MODIFIED) - Updated guidance for new standards

### Affected Files
- 📝 **Core Documentation** (5 files):
  - `00_METAFRAMEWORK_INDEX.md`
  - `01_FRAMEWORK_OVERVIEW.md`
  - `README.md`
  - `DIRECTORY_STRUCTURE.md`
  - `IMPLEMENTATION_SUMMARY.md`

- 📋 **Regulations** (1 directory removed, 4 directories added):
  - Remove: `regulations/ISO-20816/`
  - Add: `regulations/IEC-62304/` (7 templates)
  - Add: `regulations/RFC-2119/` (2 guides)
  - Add: `regulations/ISO-27001/` (3 templates)
  - Add: `regulations/ISO-9001/` (2 templates, optional)
  - Modify: `regulations/00_REGULATIONS_INDEX.md`

- 📄 **Requirements** (2 files):
  - `requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md`
  - `requirements/HOWTO_WRITE_REQUIREMENTS.md`

- 🤝 **Guides** (2 files):
  - `guides/AI_COLLABORATION_PATTERNS.md`
  - `guides/TRACEABILITY_MANAGEMENT.md` (to be created)

### Breaking Changes
- ❌ **ISO 20816 Removal**: Existing projects using ISO 20816 templates will need migration guide
- ⚠️ **Requirement Format**: Existing requirements should be updated to use RFC 2119 keywords
- ⚠️ **Traceability Matrix**: Extended to include risk controls and SOUP tracking

### Migration Path
For existing projects using ISO 20816:
1. Archive existing `regulations/ISO-20816/` content in project
2. If vibration monitoring is still needed, maintain ISO 20816 separately alongside IEC-62304
3. Update requirements to use RFC 2119 keywords (can be done incrementally)
4. Perform IEC-62304 safety classification for medical device software
5. Establish SOUP tracking for third-party libraries

## Dependencies
- None (this is a documentation-only change)

## Risks and Mitigations

### Risk 1: Existing Users Need Migration
**Impact**: Medium
**Mitigation**:
- Provide detailed migration guide
- Keep ISO 20816 as an example in archived documentation
- Create comparison table showing how concepts map between standards

### Risk 2: Increased Complexity
**Impact**: Medium
**Mitigation**:
- IEC-62304 Class A (lowest risk) projects can use simplified templates
- ISO 9001 is optional, not mandatory
- Clear decision tree for standard selection

### Risk 3: Incomplete Templates
**Impact**: Low
**Mitigation**:
- Start with core templates (SRS, SDD, STP)
- Iteratively add advanced templates based on user feedback
- Reference official IEC-62304 standard for detailed requirements

## Alternatives Considered

### Alternative 1: Keep ISO 20816, Add IEC-62304
**Rejected because**:
- Dilutes framework focus
- Increases maintenance burden
- Most users need one OR the other, not both

### Alternative 2: Create Separate Medical Device Framework
**Rejected because**:
- Violates DRY principle
- Core framework concepts (requirements, traceability, OpenSpec) are identical
- Better to have one flexible framework with standard-specific templates

### Alternative 3: Make All Standards Optional
**Rejected because**:
- Removes clear guidance
- Users need opinionated defaults
- Compliance mapping requires specific standard focus

## Success Criteria

### Functional Success
- ✅ All ISO 20816 references removed from documentation
- ✅ Complete IEC-62304 template set provided (SRS, SDD, STP, SOUP, Risk Bridge)
- ✅ RFC 2119 keyword guide integrated into requirement writing
- ✅ ISO 27001 security templates available
- ✅ ISO 9001 optional templates provided
- ✅ All existing meta-framework concepts (traceability, OpenSpec integration) work with new standards

### Quality Success
- ✅ `openspec validate migrate-to-iec62304 --strict` passes
- ✅ All spec deltas have proper ADDED/MODIFIED/REMOVED sections
- ✅ All requirements have RFC 2119 keyword examples
- ✅ Templates include realistic medical device software examples

### Documentation Success
- ✅ Migration guide available for ISO 20816 → IEC-62304
- ✅ Updated standard selection decision tree
- ✅ Compliance mapping examples for IEC-62304
- ✅ SOUP management workflow documented

## Timeline Estimate
- **Total**: 3-4 weeks for complete implementation

### Phase 1: Core Standards (Week 1-2)
- Remove ISO 20816 references
- Create IEC-62304 templates (SRS, SDD, STP)
- Add RFC 2119 guide
- Update core documentation

### Phase 2: Security & Quality (Week 2-3)
- Create ISO 27001 templates
- Create ISO 9001 optional templates
- Add SOUP management guide
- Update requirement templates

### Phase 3: Integration & Validation (Week 3-4)
- Update all guides (AI collaboration, traceability)
- Create migration guide
- Validate all templates with examples
- Final review and OpenSpec validation

## Approval Required
- [ ] Technical lead approval (architectural alignment)
- [ ] Product owner approval (business value)
- [ ] Documentation team review
- [ ] Quality assurance validation

## References
- IEC 62304:2006+A1:2015 - Medical device software lifecycle processes
- RFC 2119 - Key words for use in RFCs to Indicate Requirement Levels
- ISO/IEC 27001:2022 - Information security management systems
- ISO 9001:2015 - Quality management systems
- ISO 14971:2019 - Medical devices - Application of risk management
