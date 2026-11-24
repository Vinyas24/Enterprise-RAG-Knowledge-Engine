# Security Policy

## Passwords and multi-factor authentication

Employees must rotate passwords every 90 days and use at least 16 characters. Password managers are required for shared service accounts. Hardware security keys are mandatory for production access; SMS one-time codes are not accepted for administrative systems.

## Data encryption

Customer data is encrypted at rest with AES-256 and in transit with TLS 1.3. Encryption keys are managed by the platform KMS and rotate annually. Never copy production data to a laptop or an unapproved cloud account.

## Incident reporting

Report a lost laptop, stolen phone or suspected credential leak within one hour to security@example.com and the on-call security engineer. The security team can remotely wipe enrolled devices. Do not attempt to investigate a suspected breach alone.

## Access reviews

Managers review team access every quarter. Remove accounts for departing employees on their last working day. Production privileges are granted just in time and expire after eight hours.
