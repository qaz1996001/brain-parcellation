# Tasks: migrate-to-iec62304

## Phase 1: ISO 20816 Removal & Core Documentation Updates (Week 1-2)

### 1.1 Remove ISO 20816 References
- [x] 1.1.1 Remove empty `regulations/ISO-20816/` directory
- [x] 1.1.2 Update `00_METAFRAMEWORK_INDEX.md` - remove ISO 20816 from § 5.2
- [x] 1.1.3 Update `01_FRAMEWORK_OVERVIEW.md` - remove vibration monitoring use case
- [x] 1.1.4 Update `README.md` - remove ISO 20816 from standards list
- [x] 1.1.5 Update `regulations/00_REGULATIONS_INDEX.md` - remove § 4 (ISO 20816)
- [x] 1.1.6 Search and remove all other ISO 20816 references in documentation

### 1.2 Create IEC-62304 Compliance Framework
- [x] 1.2.1 Create `regulations/IEC-62304/` directory structure
- [x] 1.2.2 Create `compliance-mapping-template.md` - IEC-62304 compliance checklist
- [x] 1.2.3 Create `software-safety-classification.md` - Class A/B/C assessment guide with examples
- [x] 1.2.4 Create `software-requirements-spec-template.md` - SRS template aligned with IEC-62304
- [x] 1.2.5 Create `software-design-spec-template.md` - SDD template aligned with IEC-62304
- [x] 1.2.6 Create `software-test-plan-template.md` - STP template aligned with IEC-62304
- [x] 1.2.7 Create `soup-management-template.md` - SOUP tracking and evaluation template

### 1.3 Create RFC 2119 Integration
- [x] 1.3.1 Create `regulations/RFC-2119/` directory
- [x] 1.3.2 Create `requirement-keywords-guide.md` - SHALL, MUST, SHOULD, MAY definitions and usage
- [x] 1.3.3 Create `requirement-examples.md` - Medical device requirement examples using RFC 2119

### 1.4 Update Core Documentation Index
- [x] 1.4.1 Update `00_METAFRAMEWORK_INDEX.md` § 5 - Add IEC-62304, RFC 2119, ISO 27001
- [x] 1.4.2 Add software safety classification concept explanation
- [x] 1.4.3 Add SOUP management concept introduction
- [x] 1.4.4 Update traceability chain diagram to include risk controls

### 1.5 Update Framework Overview
- [ ] 1.5.1 Update `01_FRAMEWORK_OVERVIEW.md` § 2.3 - Standards compliance table
- [ ] 1.5.2 Replace vibration monitoring use case with medical device example
- [ ] 1.5.3 Update § 5.3 compliance checklist for IEC-62304
- [ ] 1.5.4 Add IEC-62304 lifecycle process overview diagram

## Phase 2: Security, Quality, and Risk Management (Week 2-3)

### 2.1 Create ISO 27001 Security Framework
- [ ] 2.1.1 Create `regulations/ISO-27001/` directory
- [ ] 2.1.2 Create `security-requirements-template.md` - Information security controls for medical data
- [ ] 2.1.3 Create `compliance-mapping-template.md` - ISO 27001 compliance checklist
- [ ] 2.1.4 Create `data-protection-guide.md` - Patient data protection requirements (HIPAA, GDPR alignment)

### 2.2 Create ISO 9001 Optional Templates
- [ ] 2.2.1 Create `regulations/ISO-9001/` directory
- [ ] 2.2.2 Create `quality-management-template.md` - QMS process documentation
- [ ] 2.2.3 Create `compliance-mapping-template.md` - ISO 9001 compliance checklist
- [ ] 2.2.4 Mark ISO 9001 as "Optional" in all documentation

### 2.3 Create Risk Management Bridge
- [ ] 2.3.1 Create `regulations/IEC-62304/risk-management-bridge-template.md`
- [ ] 2.3.2 Document IEC-62304 + ISO 14971 integration
- [ ] 2.3.3 Create risk control traceability matrix template
- [ ] 2.3.4 Add risk-based testing examples

### 2.4 Update Regulations Index
- [ ] 2.4.1 Update `regulations/00_REGULATIONS_INDEX.md` - Add § 4 (IEC-62304)
- [ ] 2.4.2 Add § 5 (RFC 2119)
- [ ] 2.4.3 Add § 6 (ISO 27001)
- [ ] 2.4.4 Add § 7 (ISO 9001 - Optional)
- [ ] 2.4.5 Update § 5.1 - Standards selection decision tree
- [ ] 2.4.6 Update § 6 - Evidence package structure for medical device submissions

### 2.5 Update Requirements Templates
- [ ] 2.5.1 Update `requirements/TEMPLATE_SYSTEM_PRD_SR_SD.md` - Add RFC 2119 usage section
- [ ] 2.5.2 Add IEC-62304 software safety class field (Class A/B/C)
- [ ] 2.5.3 Add SOUP tracking section
- [ ] 2.5.4 Update traceability matrix to include risk controls
- [ ] 2.5.5 Replace vibration examples with medical software examples (patient monitoring, diagnostic, therapeutic)

## Phase 3: Guides, Migration, and Validation (Week 3-4)

### 3.1 Update Requirement Writing Guide
- [ ] 3.1.1 Update `requirements/HOWTO_WRITE_REQUIREMENTS.md` - Add § RFC 2119 Keyword Usage
- [ ] 3.1.2 Add § IEC-62304 Requirement Classification
- [ ] 3.1.3 Add § Security Requirements (ISO 27001)
- [ ] 3.1.4 Replace all examples with medical device context
- [ ] 3.1.5 Add SOUP requirement examples

### 3.2 Update AI Collaboration Guide
- [ ] 3.2.1 Update `guides/AI_COLLABORATION_PATTERNS.md` - Change ISO 20816 references to IEC-62304
- [ ] 3.2.2 Add SOUP evaluation AI workflow
- [ ] 3.2.3 Add security requirement review checklist for AI
- [ ] 3.2.4 Add IEC-62304 safety classification AI assistance pattern

### 3.3 Create/Update Traceability Guide
- [ ] 3.3.1 Create or update `guides/TRACEABILITY_MANAGEMENT.md`
- [ ] 3.3.2 Add risk control traceability (IEC-62304 + ISO 14971)
- [ ] 3.3.3 Add SOUP traceability requirements
- [ ] 3.3.4 Add traceability examples for medical device software

### 3.4 Create Migration Guide
- [ ] 3.4.1 Create `guides/MIGRATION_ISO20816_TO_IEC62304.md`
- [ ] 3.4.2 Document conceptual mapping between standards
- [ ] 3.4.3 Provide step-by-step migration checklist
- [ ] 3.4.4 Add FAQ for migration questions

### 3.5 Update Main README
- [ ] 3.5.1 Update `README.md` § 📐 - Standards Support section
- [ ] 3.5.2 Replace Scenario 3 (vibration monitoring) with medical device scenario
- [ ] 3.5.3 Update success criteria to reflect IEC-62304 requirements
- [ ] 3.5.4 Add medical device software development use case

### 3.6 Update Supporting Documentation
- [ ] 3.6.1 Update `DIRECTORY_STRUCTURE.md` - Reflect new regulations directory structure
- [ ] 3.6.2 Update `IMPLEMENTATION_SUMMARY.md` - Note standards migration
- [ ] 3.6.3 Update `openspec-integration/00_OPENSPEC_INTEGRATION.md` - Add IEC-62304 change examples

## Phase 4: Examples, Validation, and Finalization (Week 4)

### 4.1 Create Realistic Examples
- [ ] 4.1.1 Create medical device software example in IEC-62304 SRS template
- [ ] 4.1.2 Create example SOUP evaluation (e.g., OpenSSL, React, Python libraries)
- [ ] 4.1.3 Create example risk control traceability matrix
- [ ] 4.1.4 Create example security requirements for patient data

### 4.2 Quality Validation
- [ ] 4.2.1 Run `openspec validate migrate-to-iec62304 --strict` and fix all errors
- [ ] 4.2.2 Verify all spec deltas have proper structure (ADDED/MODIFIED/REMOVED)
- [ ] 4.2.3 Check all RFC 2119 keyword usage is consistent
- [ ] 4.2.4 Verify all medical device examples are realistic and compliant

### 4.3 Cross-Reference Validation
- [ ] 4.3.1 Verify all internal links work correctly
- [ ] 4.3.2 Ensure traceability between templates is correct
- [ ] 4.3.3 Check that all removed ISO 20816 references are gone
- [ ] 4.3.4 Validate standard selection decision tree logic

### 4.4 Documentation Review
- [ ] 4.4.1 Spell check and grammar review all new content
- [ ] 4.4.2 Ensure consistent terminology (SRS, SDD, STP, SOUP)
- [ ] 4.4.3 Verify all templates have version numbers and dates
- [ ] 4.4.4 Check that optional vs. mandatory standards are clearly marked

### 4.5 Final Integration
- [ ] 4.5.1 Test end-to-end workflow: select IEC-62304 → create SRS → create SOUP list → trace to risk
- [ ] 4.5.2 Ensure OpenSpec integration still works with new templates
- [ ] 4.5.3 Verify AI collaboration patterns work with new standards
- [ ] 4.5.4 Update framework version to reflect major change

## Acceptance Criteria
- ✅ Zero references to ISO 20816 in meta-framework documentation
- ✅ Complete IEC-62304 template set (SRS, SDD, STP, SOUP, Risk Bridge)
- ✅ RFC 2119 keywords integrated into all requirement examples
- ✅ ISO 27001 security templates available
- ✅ ISO 9001 marked as optional with complete templates
- ✅ Migration guide from ISO 20816 to IEC-62304 available
- ✅ All traceability examples include risk controls
- ✅ SOUP management workflow documented
- ✅ `openspec validate` passes with no errors
- ✅ At least 3 realistic medical device software examples provided

## Notes
- This is a documentation-only change, no code implementation
- Focus on medical device software lifecycle, not medical devices themselves
- Maintain compatibility with existing OpenSpec integration
- Ensure framework remains usable for non-medical software (via ISO 29148 + RFC 2119 only)
