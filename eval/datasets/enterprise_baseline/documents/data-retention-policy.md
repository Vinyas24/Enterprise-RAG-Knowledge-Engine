# Data Retention Policy

## Customer content

Active customer documents are retained until deletion. When a customer deletes a record, it is removed from the primary database within 30 days and from encrypted backups within 90 days. Backups are restored only for disaster recovery and are never used to answer customer queries.

## Audit logs

Security and access audit logs are retained for seven years to satisfy compliance requirements. Audit logs contain actor, action, resource and timestamp, but never document content. Access to audit logs requires the security auditor role.

## Erasure requests

Handle GDPR and CCPA erasure requests within 30 days. The privacy team verifies identity, checks for a legal hold, and then deletes the content and its vectors. A confirmation is recorded in the request tracker; the requester receives written confirmation.

## Backup verification

Restore a random backup into an isolated environment every quarter and verify row counts and checksums. A failed restore test is a severity two incident. Backup encryption keys are stored separately from the backup data.
