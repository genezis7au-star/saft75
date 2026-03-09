# CLAUDE.md

This file provides guidance to AI assistants (such as Claude) working in this repository.

## Repository Overview

This is a newly initialized repository. This file will be updated as the project evolves with source code, configuration, and established conventions.

## Repository Status

- **State**: Freshly initialized — no source files committed yet
- **Branch convention**: Feature branches follow the pattern `claude/<description>-<session-id>`
- **Remote**: `genezis7au-star/saft75`

## Git Workflow

### Branch naming
- AI-assisted work: `claude/<short-description>-<session-id>`
- Features: `feature/<description>`
- Bug fixes: `fix/<description>`

### Commit messages
Write clear, imperative-style commit messages:
```
Add authentication middleware
Fix null pointer in user service
Refactor database connection pool
```

### Pushing changes
Always push with tracking:
```bash
git push -u origin <branch-name>
```

### Pull requests
- Keep PRs focused on a single concern
- Include a clear description of what changed and why

## Development Setup

> This section will be populated once the project stack is established.

Typical setup steps will be documented here, e.g.:
```bash
# Install dependencies
# Run dev server
# Run tests
```

## Project Structure

> This section will be populated as the codebase grows.

Expected layout will be documented here once directories and files are established.

## Testing

> Document test commands here once a testing framework is configured.

```bash
# Run all tests
# Run a single test file
# Run with coverage
```

## Code Conventions

> These will be refined once the project language and framework are chosen.

General principles to follow regardless of stack:
- Prefer clarity over cleverness
- Keep functions small and focused on a single responsibility
- Avoid over-engineering — build only what is needed now
- Do not add error handling for scenarios that cannot occur
- Do not add comments unless the logic is non-obvious

## Security

- Never commit secrets, API keys, or credentials
- Add `.env` and secret files to `.gitignore` before first commit
- Validate all user input at system boundaries

## AI Assistant Guidelines

When working in this repository, AI assistants should:

1. **Read before editing** — always read a file before modifying it
2. **Stay in scope** — only change what was requested; avoid unsolicited refactors
3. **Keep changes minimal** — prefer editing existing files over creating new ones
4. **Confirm before destructive actions** — force pushes, file deletions, branch resets require explicit user approval
5. **Use the designated branch** — all work goes to the branch specified in the task context
6. **Update this file** — when new conventions, tools, or structure are established, update the relevant section of this file

## Updating This File

As the project grows, update this file to reflect:
- The actual tech stack and language
- Real install/run/test commands
- Confirmed code style and linting rules
- CI/CD pipeline details
- Any domain-specific conventions
