# Specification: ISO 27001 Information Security

## ADDED Requirements

### Requirement: Information Security Management System (ISMS) Support
The meta-framework SHALL provide templates for integrating ISO/IEC 27001:2022 information security requirements into medical device software development.

#### Scenario: Security Requirements Definition
- **GIVEN** a medical device software that processes patient data
- **WHEN** defining security requirements
- **THEN** the framework SHALL provide templates for:
  - Confidentiality requirements (encryption, access control)
  - Integrity requirements (data validation, audit trails)
  - Availability requirements (redundancy, disaster recovery)
  - Authentication and authorization requirements
  - Data protection requirements (GDPR, HIPAA alignment)

### Requirement: Security Control Mapping
The meta-framework SHALL provide a security requirements template that maps to ISO 27001 Annex A controls.

#### Scenario: Security Control Implementation Planning
- **GIVEN** ISO 27001 security control requirements
- **WHEN** planning security implementation
- **THEN** the template SHALL map medical device software requirements to:
  - A.5: Organizational controls
  - A.6: People controls
  - A.7: Physical controls
  - A.8: Technological controls (primary focus for software)
- **AND** technical controls SHALL include specific guidance for:
  - Access control (A.8.2-A.8.5)
  - Cryptography (A.8.24)
  - Secure development lifecycle (A.8.25-A.8.31)
  - Logging and monitoring (A.8.15-A.8.16)

### Requirement: Patient Data Protection Requirements
The meta-framework SHALL provide specific guidance for protecting patient health information (PHI) in compliance with data protection regulations.

#### Scenario: PHI Protection Implementation
- **GIVEN** software that stores or processes patient health information
- **WHEN** implementing data protection
- **THEN** security requirements SHALL address:
  - Encryption at rest (AES-256 or equivalent)
  - Encryption in transit (TLS 1.2+ or equivalent)
  - Access control (RBAC, principle of least privilege)
  - Audit logging (who accessed what data when)
  - Data minimization (collect only necessary data)
  - Data retention and deletion policies
  - Breach notification procedures
- **AND** requirements SHALL align with GDPR (EU), HIPAA (US), and local regulations

### Requirement: Security Risk Assessment Integration
The meta-framework SHALL integrate security risk assessment with IEC-62304 software safety risk management.

#### Scenario: Combined Safety and Security Risk Analysis
- **GIVEN** a medical device software component
- **WHEN** performing risk analysis
- **THEN** the framework SHALL support evaluation of:
  - Safety risks (harm to patients from software failures)
  - Security risks (harm to patients from security breaches)
  - Combined safety-security risks (e.g., ransomware affecting life-support software)
- **AND** security controls SHALL be traceable to security risks
- **AND** security risks SHALL be considered in IEC-62304 safety classification

### Requirement: Secure Development Lifecycle Requirements
The meta-framework SHALL provide secure coding and development process requirements aligned with ISO 27001 A.8.25-A.8.31.

#### Scenario: Secure Development Process Definition
- **GIVEN** a medical device software development project
- **WHEN** establishing development processes
- **THEN** security requirements SHALL cover:
  - Secure coding standards (OWASP, CERT, CWE Top 25)
  - Code review and static analysis
  - Security testing (penetration testing, vulnerability scanning)
  - Dependency management (SOUP security evaluation)
  - Secrets management (no hardcoded credentials)
  - Security patch management process
- **AND** requirements SHALL be integrated with IEC-62304 development lifecycle

### Requirement: Incident Response and Logging Requirements
The meta-framework SHALL provide templates for security event logging and incident response requirements.

#### Scenario: Security Event Logging
- **GIVEN** a medical device software in operation
- **WHEN** security-relevant events occur
- **THEN** the system SHALL log:
  - Authentication events (login, logout, failed attempts)
  - Authorization events (access grants, access denials)
  - Data access events (who accessed which patient records)
  - System configuration changes
  - Security policy changes
- **AND** logs SHALL be tamper-evident and retained per regulatory requirements
- **AND** logs SHALL support forensic analysis and breach notification

## MODIFIED Requirements

### Requirement: Traceability Matrix Extension (Security Controls)
~~The meta-framework SHALL provide traceability from requirements through implementation to test cases.~~

**UPDATED**: The traceability matrix SHALL additionally track:
- Security threats to security requirements
- Security requirements to security controls
- Security controls to implementation
- Security controls to verification (security tests)
- Security controls to ISO 27001 Annex A controls

#### Scenario: Security Audit Traceability
- **GIVEN** a security audit of medical device software
- **WHEN** demonstrating security control implementation
- **THEN** traceability SHALL show:
  - Identified security threats (e.g., SQL injection, unauthorized access)
  - Security requirements addressing threats
  - Implemented security controls (e.g., parameterized queries, authentication)
  - Test cases verifying controls
  - Mapping to ISO 27001 controls (e.g., A.8.3 Access Control)

### Requirement: SOUP Security Evaluation (Extended)
~~The meta-framework SHALL provide SOUP (third-party software) management templates.~~

**UPDATED**: SOUP evaluation SHALL additionally include:
- Known security vulnerabilities (CVE database checks)
- Security patch availability and cadence
- Security-specific verification testing
- Security implications for IEC-62304 safety classification

#### Scenario: SOUP Security Assessment
- **GIVEN** a third-party library to be integrated (e.g., OpenSSL, React)
- **WHEN** performing SOUP evaluation
- **THEN** the assessment SHALL include:
  - Search for known CVEs in NIST NVD database
  - Review of supplier's security update history
  - Assessment of security configuration requirements
  - Security testing plan for the integrated component
- **AND** high-severity security vulnerabilities SHALL impact safety classification

## Dependencies
- Integrates with IEC-62304 for combined safety-security risk management
- Supports RFC 2119 keywords for security requirement precision
- References GDPR, HIPAA for data protection alignment

## Validation
- Security templates SHALL cover all applicable ISO 27001 Annex A controls
- Data protection guidance SHALL align with GDPR and HIPAA requirements
- Security risk examples SHALL demonstrate safety-security interactions
- Traceability examples SHALL show complete threat-to-test chains
