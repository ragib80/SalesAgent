Please read the `docs/overview.md` and `docs/phase_history.md` carefully before making any code changes. The README already contains the complete project architecture, implementation roadmap, technical requirements, and phase-wise migration plan.

Our objective is to **refactor the existing project incrementally**, implementing one phase at a time while preserving current functionality. Do not introduce large-scale changes that impact existing behavior unless explicitly required by the current phase.

## Implementation Guidelines

1. Thoroughly analyze the existing codebase and README before making changes.
2. Follow the implementation roadmap exactly as documented in the README.
3. Implement only the current phase being worked on.
4. Maintain backward compatibility with the existing application.
5. Refactor existing code where necessary, but avoid unnecessary rewrites.
6. Keep the code modular, maintainable, and production-ready.
7. Follow established design patterns and best practices.
8. Add clear comments and documentation for any new components.
9. Ensure each phase is fully tested and functional before moving to the next phase.
10. At the end of each phase, provide:

    * Summary of changes
    * Files modified
    * New files created
    * Migration considerations
    * Next phase recommendations

## Environment Variables

The following Azure Cognitive Search configuration is already available in the `.env` file and must be used instead of hardcoded values:

```env
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX_NAME=
```

## Phase History Tracking

Create and maintain a project-level implementation history file to track all completed work throughout the refactoring process.

### Required File

Create:

```text
docs/phase_history.md
```

### Purpose

This file will serve as the single source of truth for implementation progress and project evolution.

### After Completing Each Phase

Update `docs/phase_history.md` with:

* Phase number and title
* Completion date
* Objective of the phase
* Architecture changes
* Features implemented
* Refactoring performed
* Files added
* Files modified
* Breaking changes (if any)
* Migration steps (if any)
* Testing performed
* Known issues or limitations
* Lessons learned
* Next phase recommendations

### Example Structure

```markdown
# Phase Implementation History

## Phase 1 - LangGraph Foundation

### Date
YYYY-MM-DD

### Objective
Brief description.

### Changes Implemented
- Item 1
- Item 2

### Files Added
- file1.py
- file2.py

### Files Modified
- existing_file.py

### Testing
- Unit tests passed
- Integration tests passed

### Notes
Additional information.

### Next Phase
Planned activities for the next phase.
```

The file must be updated at the end of every completed phase so that any developer can understand the project's evolution without reviewing the entire commit history.

## Expected Approach

* Review the README and understand the complete architecture.
* Analyze the current implementation and identify the components related to the current phase.
* Create a detailed implementation plan before modifying code.
* Implement the phase incrementally.
* Preserve all existing functionality unless the phase specifically requires changes.
* Follow the roadmap exactly as described in the README.

## Before Writing Any Code

Provide the following:

1. Summary of your understanding of the current architecture.
2. Scope of the current phase.
3. Detailed implementation plan.
4. Potential risks and mitigation strategies.
5. Files that will be modified.
6. Files that will be added.
7. How the phase history file (`docs/phase_history.md`) will be updated.

Only after presenting the plan and receiving confirmation should implementation begin.




## Python & Django Clean Code Standards

All new code and refactored code must follow Python and Django best practices.

### General Principles

* Follow PEP 8 coding standards.
* Prefer readability over clever implementations.
* Keep functions small and focused on a single responsibility.
* Avoid code duplication (DRY principle).
* follow the SOLID principle.
* Use meaningful and descriptive names for variables, functions, classes, and modules.
* Remove dead code, commented-out code, and unused imports.
* Use type hints wherever practical.
* Add docstrings for public classes, methods, and complex functions.
* Keep business logic separate from presentation and infrastructure layers.

### Django Architecture

* Follow Django best practices and maintain a clear separation of concerns.
* Keep views/controllers thin.
* Move business logic into dedicated service layers.
* Keep database access isolated in repositories, managers, or service classes where appropriate.
* Avoid placing complex business logic directly inside views, serializers, models, or signals.
* Use Django settings and environment variables for configuration.
* Never hardcode secrets, endpoints, keys, or configuration values.

### Project Structure

* Organize code into logical modules and packages.
* Group related functionality together.
* Maintain a scalable folder structure suitable for enterprise applications.
* Ensure new components align with the existing architecture defined in the README.

### Error Handling & Logging

* Implement proper exception handling.
* Avoid generic `except Exception` blocks unless absolutely necessary.
* Log meaningful errors and important application events.
* Provide actionable error messages.
* Do not expose sensitive information in logs.

### Database & Query Optimization

* Avoid N+1 query problems.
* Use `select_related()` and `prefetch_related()` where appropriate.
* Optimize database queries before introducing caching.
* Keep ORM queries efficient and readable.
* Add comments for complex query logic.

### API Development

* Keep API endpoints focused and consistent.
* Validate all incoming requests.
* Use serializers/schema validation properly.
* Return standardized response structures.
* Handle edge cases and invalid inputs gracefully.

### Testing

* Add tests for all new business logic.
* Ensure existing functionality remains intact.
* Prefer automated tests over manual verification.
* Update existing tests when refactoring behavior.

### Documentation

* Document architectural decisions.
* Document new services, workflows, and integrations.
* Update README and `docs/phase_history.md` when changes are introduced.
* Include implementation notes for future developers.

### Refactoring Rules

* Refactor incrementally.
* Preserve existing functionality unless the current phase explicitly requires behavioral changes.
* Avoid large-scale rewrites.
* Make small, verifiable improvements with each phase.
* Ensure every refactor improves maintainability, readability, or performance.

### Code Review Checklist

Before considering a phase complete, verify:

* PEP 8 compliant
* No unused imports
* No duplicated logic
* Proper type hints added
* Appropriate docstrings added
* Error handling implemented
* Logging added where necessary
* Tests updated or created
* Documentation updated
* `docs/phase_history.md` updated
* Existing functionality verified
