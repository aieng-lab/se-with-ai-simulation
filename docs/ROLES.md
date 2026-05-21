# ROLES.md

This document defines the roles of all participants in a mixed AI-agent/human-developer team. It serves as the foundation for the simulation model: each role represents a distinct actor whose behavior will be characterized by random variables in subsequent modeling phases.

The role model distinguishes between **AI Agents** — autonomous agents driven by large language models that execute tasks with measurable cost, speed, and quality characteristics — and **Human Developers** — people who provide judgment, oversight, and skills that agents currently cannot reliably replicate.

The human role model is designed to scale: rather than assuming singleton roles (e.g., one technical lead, one manager), the model assumes a human team where each human developer oversees a sub-team of agents. The key simulation variable for human roles is the **span of control** — how many agents a single human can effectively supervise. Team size is determined by how many humans are employed and how many agents each oversees.

---

## AI Agent Roles

### 1. Requirements Refinement Agent

**Purpose:**  
Takes raw, ambiguous, or informal user requirements and transforms them into clear, structured, and unambiguous specifications. Identifies gaps, contradictions, and underspecified behavior. Produces artifacts that all downstream roles (architecture, implementation, testing) can act on without further clarification.

**Primary Inputs:**
- Raw user requirements or feature requests (from Product Owner)
- Existing system documentation or domain context

**Primary Outputs:**
- Structured requirement specifications (e.g., user stories, functional specs, acceptance criteria)
- Lists of open questions or ambiguities requiring human clarification

**Key Interactions:**
- Receives raw requirements from the **Product Owner**
- Receives confirmation for refined specifications from **Product Owner**
- Hands off refined specifications to the **Architecture and Design Proposal Agent** and **Feature Implementation Agent**
- Escalates unresolvable ambiguities to the **Product Owner**

**Simulation Notes:**  
Produces a defined number of requirements per unit time. Quality of output affects downstream defect rates (poorly refined requirements lead to misimplemented features). May introduce "specification bugs" — misinterpretations that propagate to implementation.

---

### 2. Architecture and Design Proposal Agent

**Purpose:**  
Designs the high-level structure of the system or a new subsystem based on refined requirements. Produces architectural proposals including component decomposition, interface definitions, data models, technology choices, and interaction patterns. Outputs are proposals, not final decisions — the Human Developer reviews and approves them.

**Primary Inputs:**
- Refined requirement specifications (from Requirements Refinement Agent)
- Existing system architecture documentation
- Constraints (technology stack, performance requirements, budget)

**Primary Outputs:**
- Architecture and design documents (component diagrams, API contracts, data models, ADRs)
- Design alternatives with trade-off analysis

**Key Interactions:**
- Receives requirements from the **Requirements Refinement Agent**
- Proposals reviewed and approved by the **Human Developer**
- Approved designs consumed by the **Feature Implementation Agent**, **DevOps/CI-CD Agent**, and test agents

**Simulation Notes:**  
Design quality significantly affects downstream implementation quality and maintenance costs. Poor architecture decisions may multiply bug rates or increase refactoring needs. Has a relatively low per-token cost but potentially high time cost for complex systems.

---

### 3. Feature Implementation Agent

**Purpose:**  
Implements new features in the codebase according to approved design specifications and refined requirements. Writes production code that is expected to be functionally correct, readable, and consistent with the existing codebase style. The primary source of new functionality — and also a primary source of newly introduced bugs.

**Primary Inputs:**
- Refined requirement specifications (from Requirements Refinement Agent)
- Approved architecture and design documents (from Architecture and Design Proposal Agent)
- Existing codebase

**Primary Outputs:**
- New or modified production code
- Basic inline documentation (code comments, docstrings)

**Key Interactions:**
- Works from specs produced by the **Requirements Refinement Agent** and **Architecture and Design Proposal Agent**
- Code reviewed by the **Code Review Agent** and **Security Audit Agent**
- Escalates especially complex implementation problems to the **Human Developer**

**Simulation Notes:**  
The central productivity driver. Key variables: features implemented per unit time, bugs introduced per feature (defect injection rate), token cost per feature. Bug injection rate is influenced by requirements quality, design quality and the internal code quality of the code base (technical debt).

---

### 4. Bug Fixing Agent

**Purpose:**  
Diagnoses and resolves reported bugs in the codebase. Performs root cause analysis, identifies the defective code, implements a fix, and verifies the fix does not introduce regressions. Distinct from Feature Implementation in that the input is a defect report rather than a feature specification, and the goal is restoring correct behavior rather than adding new behavior.

**Primary Inputs:**
- Bug reports with reproduction steps and expected vs. actual behavior
- Relevant sections of the codebase

**Primary Outputs:**
- Bug fixes (code patches)
- Root cause analysis notes

**Key Interactions:**
- Receives bug reports surfaced by test agents (**Unit Test**, **Integration Test**, **System and Acceptance Test**, **Security Audit**)
- Receives bug reports from human developers
- Fixes reviewed by the **Code Review Agent**
- May escalate complex bugs to the **Human Developer**

**Simulation Notes:**  
Key variables: bugs fixed per unit time, probability of introducing a new bug while fixing (fix-induced defect rate), time per fix (varies with bug complexity). The fix-induced defect rate distinguishes this role from Feature Implementation.

---

### 5. Refactoring Agent

**Purpose:**  
Improves the internal structure, readability, and maintainability of existing code without altering external behavior. Addresses technical debt, reduces complexity, improves naming, eliminates duplication, and restructures modules. Does not add new features or fix functional bugs. Does not modify the architecture.

**Primary Inputs:**
- Sections of the codebase identified as needing improvement
- Code quality metrics or refactoring requests (from Code Review Agent or Human Developer)

**Primary Outputs:**
- Refactored code (functionally equivalent, structurally improved)
- Refactoring notes describing changes made

**Key Interactions:**
- Triggered by recommendations from **Code Review Agent** or **Human Developer**
- Refactored code validated by the **Unit Test Development Agent** (regression risk)
- Output reviewed by the **Code Review Agent**

**Simulation Notes:**  
Refactoring has near-zero direct feature output but reduces future bug rates and implementation time by decreasing codebase complexity. Models the trade-off between investing in quality vs. shipping speed. Has a non-zero bug introduction rate (regressions), but improved code-quality positively affects speed and quality of introducing new features.

---

### 6. Code Review Agent

**Purpose:**  
Reviews code produced by other agents (primarily Feature Implementation, Bug Fixing, and Refactoring) for correctness, style, maintainability, adherence to design, and potential functional defects. Provides structured feedback and either approves or requests changes.

**Primary Inputs:**
- Code changes (diffs, pull requests) from Feature Implementation, Bug Fixing, or Refactoring agents
- Approved design documents and coding standards

**Primary Outputs:**
- Review feedback (approval, change requests, comments)
- Identified functional defects and code quality issues

**Key Interactions:**
- Reviews output from **Feature Implementation Agent**, **Bug Fixing Agent**, and **Refactoring Agent**
- Routes identified bugs to the **Bug Fixing Agent**
- Escalates architectural concerns to the **Human Developer**
- Works in parallel with the **Security Audit Agent** (complementary, not redundant)

**Simulation Notes:**  
Key variables: bug detection rate (fraction of existing bugs caught per review), false positive rate, review time per change. Bug detection here prevents bugs from reaching testing or production. Detection rate may decrease as change size increases.

---

### 7. Security Audit Agent

**Purpose:**  
Performs security-focused review of code and system design. Identifies vulnerabilities (injection, authentication flaws, insecure dependencies, data exposure, etc.), assesses threat exposure, and recommends remediations. Complements the Code Review Agent with domain-specific security expertise.

**Primary Inputs:**
- Code changes or full codebase sections
- Dependency manifests
- Architecture and design documents

**Primary Outputs:**
- Security findings with severity ratings
- Remediation recommendations
- Approved security posture assessment

**Key Interactions:**
- Reviews output from **Feature Implementation Agent** and **Architecture and Design Proposal Agent**
- Routes security defects to the **Bug Fixing Agent**
- Reports critical findings to the **Human Developer**

**Simulation Notes:**  
Distinct from Code Review Agent because it finds a different class of defects (security bugs) at a different detection rate. Security bugs may have asymmetric cost impact (e.g., data breaches). Models the trade-off of investing in security review vs. speed.

---

### 8. Unit Test Development Agent

**Purpose:**  
Writes automated unit tests that verify the correctness of individual functions, classes, and modules in isolation. Aims for high coverage of the logic implemented by the Feature Implementation and Bug Fixing agents. Unit tests are the first line of automated defect detection.

**Primary Inputs:**
- Production code to be tested (from Feature Implementation Agent)
- Requirement specifications (for understanding expected behavior)

**Primary Outputs:**
- Unit test suites (per module or feature)
- Test coverage reports

**Key Interactions:**
- Tests code produced by **Feature Implementation Agent** and **Bug Fixing Agent**
- Detected bugs routed to **Bug Fixing Agent**
- Test suites maintained alongside refactoring by **Refactoring Agent**

**Simulation Notes:**  
Key variables: bug detection rate at unit level, tests written per unit time, coverage achieved, cost per test suite. Unit tests catch bugs early (low fix cost) but have limited scope (cannot catch integration or systemic issues).

---

### 9. Integration Test Development Agent

**Purpose:**  
Writes automated tests that verify the correct interaction between components, services, or modules. Focuses on interface contracts, data flows across boundaries, and behavior of the system when components are combined — defects that unit tests cannot detect.

**Primary Inputs:**
- API contracts and interface definitions (from Architecture and Design Proposal Agent)
- Component-level production code (from Feature Implementation Agent)
- Unit test suites (as a baseline)

**Primary Outputs:**
- Integration test suites
- Identified integration defects (interface mismatches, data contract violations)

**Key Interactions:**
- Tests interactions between modules implemented by **Feature Implementation Agent**
- Detected bugs routed to **Bug Fixing Agent**
- Depends on architecture documents from **Architecture and Design Proposal Agent**

**Simulation Notes:**  
Detects a class of bugs that survive unit testing (integration defects). Higher cost per bug found than unit tests, but prevents bugs from reaching system-level testing. Bug detection rate is complementary to, not overlapping with, unit and system tests.

---

### 10. System and Acceptance Test Development Agent

**Purpose:**  
Writes end-to-end tests that validate the full system against user-facing requirements and acceptance criteria. Tests the system as a whole from a user perspective, including workflows, performance characteristics, and compliance with acceptance criteria defined by the Product Owner.

**Primary Inputs:**
- Refined requirement specifications and acceptance criteria (from Requirements Refinement Agent and Product Owner)
- Deployed or integrated system

**Primary Outputs:**
- System and acceptance test suites
- Test execution reports
- Go/no-go signal for release readiness

**Key Interactions:**
- Validates against acceptance criteria defined by the **Product Owner**
- Detected bugs routed to **Bug Fixing Agent**
- Reports release readiness to the **Product Owner** and **Human Developer**

**Simulation Notes:**  
Highest-level defect detection. Catches bugs that survive all earlier stages. Most expensive per test execution. Failure here has the highest delay cost. Provides the final quality gate before release.

---

### 11. DevOps / CI-CD Agent

**Purpose:**  
Designs, implements, and maintains the build, test automation, containerization, and deployment pipeline. Ensures that code can be reliably built, tested, and deployed across environments. Manages infrastructure-as-code, environment configuration, and release automation.

**Primary Inputs:**
- Architecture and design documents (from Architecture and Design Proposal Agent)
- Application code and test suites
- Infrastructure requirements

**Primary Outputs:**
- CI/CD pipeline configuration
- Containerization artifacts (Dockerfiles, Kubernetes manifests, IaC scripts)
- Deployment automation scripts
- Environment configuration

**Key Interactions:**
- Works from architecture defined by **Architecture and Design Proposal Agent**
- Enables automated execution of tests from all three test agents
- Coordinates with **Human Developer** on deployment schedules and release automation
- Infrastructure decisions reviewed and approved by the **Human Developer**

**Simulation Notes:**  
One-time and maintenance cost that is often overlooked in productivity models. Affects how quickly bugs are detected after introduction (pipeline speed) and deployment frequency. Models the operational overhead of the development process.

---

### 12. Developer Documentation Agent

**Purpose:**  
Produces technical documentation aimed at developers: API references, architecture guides, onboarding documentation, code-level explanations, and contributor guides. Reduces the cognitive load on human developers when overseeing agents and understanding the system.

**Primary Inputs:**
- Production code (for API docs and code-level docs)
- Architecture documents (from Architecture and Design Proposal Agent)
- Requirement specifications

**Primary Outputs:**
- API reference documentation
- Architecture and design guides
- Developer onboarding and contribution guides

**Key Interactions:**
- Documents code produced by **Feature Implementation Agent**
- Documents architecture from **Architecture and Design Proposal Agent**
- Outputs consumed by **Human Developers** for oversight and onboarding

**Simulation Notes:**  
Does not directly affect bug rates but reduces human oversight time (agents and humans can act on documented code faster, reducing per-task time). Models the indirect productivity value of documentation investment.

---

### 13. User Documentation Agent

**Purpose:**  
Produces documentation aimed at end users: user manuals, tutorials, API consumer guides, release notes, and help content. Translates technical functionality into user-facing language aligned with how users think about the product.

**Primary Inputs:**
- Refined requirement specifications and acceptance criteria (from Requirements Refinement Agent)
- System and Acceptance Test results (as a source of truth for actual behavior)
- Release scope decisions (from Product Owner)

**Primary Outputs:**
- User manuals and tutorials
- Release notes
- Help and FAQ content

**Key Interactions:**
- Receives feature scope from **Product Owner**
- Validates documented behavior against **System and Acceptance Test Development Agent** outputs
- Output reviewed by **Product Owner** for accuracy against user expectations

**Simulation Notes:**  
Does not affect defect rates but contributes to release cost and time. Its cost scales with the number of features shipped per release. Models the non-engineering overhead of each release.

---

## Human Roles

### 14. Product Owner

**Purpose:**  
The human interface between external user needs and the development team. Provides user requirements, prioritizes the backlog, defines acceptance criteria, and decides what constitutes a shippable release. Resolves requirement ambiguities that agents cannot determine autonomously. Multiple Product Owners may exist in larger organizations, each responsible for a distinct product area or feature domain.

**Primary Inputs:**
- User needs, market feedback, and business goals (external)
- Open questions from the **Requirements Refinement Agent**
- Release readiness reports from **System and Acceptance Test Development Agent**

**Primary Outputs:**
- Raw feature requests and user stories
- Prioritized backlog
- Release decisions (which features and bug fixes are in scope)
- Acceptance criteria approval

**Key Interactions:**
- Feeds requirements to the **Requirements Refinement Agent**
- Approves outputs of the **User Documentation Agent**
- Receives release readiness signals from the **System and Acceptance Test Development Agent**
- Coordinates release scope with **Human Developers**

**Simulation Notes:**  
Time spent: requirements provision, backlog prioritization, acceptance criteria definition, release gate review. Availability is a potential bottleneck (agent pipeline may block waiting for clarification). The number of Product Owners scales with the breadth of the product, not directly with the number of agents. Models the human judgment cost of product decisions.

---

### 15. Human Developer

**Purpose:**  
The core scalable human role in the team. Each Human Developer oversees a sub-team of AI agents, provides the judgment and expertise that agents cannot reliably supply, and serves as the local technical authority and escalation point for their agents. Responsibilities span: reviewing and approving architecture and design proposals, resolving escalations when agents are stuck, implementing especially complex logic directly, reviewing critical outputs that require human judgment, and allocating and monitoring work across their assigned agents.

There is no fixed specialization — the same role covers what might classically be called "tech lead", "senior developer", or "team lead". The simulation models the team as N Human Developers, each overseeing M agents, where M is the span of control.

**Primary Inputs:**
- Architecture and design proposals (from Architecture and Design Proposal Agent)
- Escalated problems from any assigned agent (Feature Implementation, Bug Fixing, Code Review, Security Audit, etc.)
- Backlog priorities from the **Product Owner**
- Status and progress signals from their supervised agents

**Primary Outputs:**
- Approved or rejected architecture decisions (with written rationale)
- Complex code contributions (for tasks beyond agent capability)
- Technical guidance for unblocking agents
- Work assignments and task allocations for their agent sub-team
- Status reports to the **Product Owner**

**Key Interactions:**
- Reviews proposals from **Architecture and Design Proposal Agent**
- Receives escalations from any agent in their sub-team
- Feeds priorities from the **Product Owner** into agent task assignments
- Collaborates with other **Human Developers** on cross-team decisions

**Simulation Notes:**  
The central variable is **span of control** (M): how many agents one Human Developer can effectively supervise. This determines the human headcount required for a given agent pool. Time is split across oversight, reviews, escalation resolution, and direct code contribution. As M increases, oversight quality degrades (higher per-agent defect escape rate, slower escalation response). The optimal M is a key output of the simulation. Human Developers have a fixed salary cost (vs. agents' per-token cost), making the human-to-agent ratio a primary cost lever.
