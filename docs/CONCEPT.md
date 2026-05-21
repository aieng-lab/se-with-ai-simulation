In the future, it is save to assume that development teams are a combination of human developers and AI Agents. 

To start this project, we assume that there are the following roles for AI Agents:
- Requirements Refinement Agent
- Architecture and Design Propsal Agent
- Feature Implementation Agent
- Bug Fixing Agent
- Refactoring Agent
- Code Review Agent
- Unit Test Development Agent
- Integration Test Development Agent
- System and Acceptance Test Development Agent
- Developer Documentation Agent
- User Documentation Agent

In this setting, Human Developers have the following tasks:
- Oversee work of multiple agents and aid them when they cannot solve a task
- Review important design and architecture decion
- Provide user requirements
- Prioritize the work
- Decide when to ship releases in which features need to be implemented and which bugs need to be fixed first

We assume that developers sometimes need to still check code themselves, including writing code for especially complex parts of the logic themselves. 

The rough plan of the project is the following:
- Review and finalize roles of agentic and human developers. The result of this should be a detailed "Roles.md" file. 
- Define which variables need to be modelled, for both agents and humans. Initial consideration are costs, time spent on different tasks, features implemented, bugs introduced, bugs found, and bugs fixed. The result of this should be a detailed "Variables.md" file.
- Develop a formal framework that models agents and human developers through random variables for the above considerations. The result of this should be a detailed "Framework.md" file.
- Develop a simulation model in Python that can run monte-carlo simulations to determine suitable team configurations with different human and agentic actors that can be used to determine how different choices affect development speed (features implemented), software quality (bugs in the software), and costs. The goal is to find pareto-optimal choices for a given budget. 
