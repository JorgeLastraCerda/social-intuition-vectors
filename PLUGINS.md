# Claude Code Plugin Routing

## General routing

At the start of each task, infer the user's intent and automatically select
the smallest relevant set of installed skills, agents, MCP tools, hooks, and
LSP tools.

Do not require the user to name plugins.

Use plugins only when the task clearly matches their purpose. Use at most:

- One primary development workflow
- One primary review workflow
- Supporting tools required for implementation or verification

Prefer this order:

1. Investigate and plan
2. Implement
3. Test and verify
4. Review
5. Simplify when useful

Do not repeatedly announce plugin usage. Mention it only for substantial
workflows, external services, or actions with side effects.

If a plugin or MCP server is unavailable, briefly report it and continue with
built-in tools when safe.

## Development workflow

Use Superpowers as the primary workflow for non-trivial development.

Use:

- `superpowers:brainstorming` for unclear requirements, architecture choices,
  or substantial new features.
- `superpowers:writing-plans` for multi-step implementations.
- `superpowers:test-driven-development` when behavior can reasonably be tested.
- `superpowers:systematic-debugging` for bugs, failing tests, build failures,
  regressions, and unexpected behavior.
- `superpowers:verification-before-completion` before claiming completion.

Skip full planning for typos, copy edits, formatting-only work, obvious
one-line fixes, or explanation-only requests.

## Frontend and browser work

Use `frontend-design` for:

- New or redesigned pages and components
- Layout, typography, responsive behavior, animation, and visual polish
- Translating a visual design into a production interface

Do not use it for backend-only work or trivial CSS fixes.

Use Playwright when browser execution is needed to verify:

- User flows, forms, navigation, and interactions
- Responsive or browser-specific behavior
- End-to-end tests
- Screenshots or visual confirmation

Do not use Playwright when static inspection, unit tests, or type checking are
sufficient.

## Documentation and diagnostics

Use Context7 when the task depends on current or version-specific third-party
documentation, unfamiliar package APIs, migrations, or deprecations.

Prefer repository code and local documentation for project-specific behavior.

For TypeScript and JavaScript changes, use TypeScript LSP diagnostics and the
project's existing type-check command when available. Do not hide errors with
`any`, unsafe assertions, or ignored diagnostics.

Allow security-guidance hooks to run automatically. Apply extra security
scrutiny to authentication, authorization, credentials, user input, database
queries, file access, HTML rendering, shell commands, payments, and secrets.

## GitHub

Use GitHub MCP for remote GitHub information such as issues, pull requests,
discussions, repository metadata, and remote branch state.

Use local git commands for local status, diffs, history, and branches.

Never create, modify, comment on, close, approve, merge, or otherwise change
remote GitHub content unless the user explicitly requests it.

## Reviews

Use `code-review` as the default review workflow for ordinary pull-request
reviews and ready-to-ship changes.

Use `pr-review-toolkit:review-pr` instead when:

- The user requests a deep or exhaustive review
- The change affects authentication, payments, permissions, migrations,
  persistence, concurrency, or public APIs
- A targeted review of tests, error handling, types, comments, or complexity
  is needed

Do not run both by default. Run both only when the user explicitly asks for a
second opinion, and report only new findings or meaningful disagreements.

Validate review findings before changing code. Never apply all recommendations
automatically.

## Simplification

Use `code-simplifier` after substantial implementation when the changed code
contains meaningful duplication, unnecessary complexity, inconsistent
patterns, or excessive nesting.

Do not use it for tiny fixes, migrations, generated code, or sensitive
security changes unless simplification is clearly safe.

Rerun relevant verification after simplification.

## Claude Code configuration

Use:

- `claude-md-management` for CLAUDE.md and memory configuration
- `skill-creator` for creating or improving skills
- `claude-code-setup` for repository-specific Claude Code configuration advice

Do not invoke these during ordinary application development.

## Side effects

Do not automatically commit, push, open pull requests, deploy, publish, merge,
write to databases, or modify remote services.

Use commit-commands only when the user explicitly asks for the corresponding
Git or GitHub action.

Before external or irreversible actions, verify the intended scope.

## Completion

Before claiming completion:

- Inspect the final diff
- Run the narrowest relevant tests
- Run type checking or compilation when applicable
- Use browser verification only when relevant
- Check for unrelated or unexpected changes
- Report what was verified and what could not be verified

Do not claim success from inspection alone when executable verification is
available.