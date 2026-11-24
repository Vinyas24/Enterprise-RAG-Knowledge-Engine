# Release Process

## Versioning

Releases follow semantic versioning. Every changelog entry includes the ticket number, a user-visible summary and migration notes. Breaking API changes require a major version and a deprecation window of at least 90 days.

## QA gate

A release candidate must pass unit tests, integration tests against PostgreSQL and the end-to-end query suite. Load tests run for 30 minutes at twice the expected peak. Security scanning and dependency review must be green before approval.

## Feature flags

New features ship behind a flag that defaults to off. Enable the flag for internal users, then a small customer cohort, then everyone. Remove the flag within 30 days of full rollout and delete the dead branch.

## Release candidate approval

Tag the release candidate, deploy it to staging, and run the smoke tests. Two engineers approve the production change, including one from the owning team. The release manager records the approval and the rollback plan in the change ticket.
