# Deployment Runbook

## Pre-flight checks

Before every production deploy, confirm the release candidate passed the QA gate and that the database migration is backward compatible. Run `kubectl diff` against the staging manifest and verify that the canary pool has capacity. Freeze deploys during an active severity one incident.

## Canary rollout

Send 5 percent of traffic to the canary pool for 15 minutes. Watch error rate, p95 latency, CPU saturation and queue depth on the deployment dashboard. If the canary is healthy, increase traffic in steps of 25 percent with a five minute observation window between steps.

## Automatic rollback

If the canary error rate exceeds 2 percent or p95 latency doubles for five minutes, the deploy controller triggers an automatic rollback. Engineers can also roll back manually with `kubectl rollout undo deployment/api`. The rollback window is 30 minutes after full rollout; after that, use a forward fix.

## Post-deploy verification

Check pod health with `kubectl get pods -n production` and confirm that every replica is ready. Run the smoke test suite against the public endpoint, verify the status page, and post the deploy summary in the release channel. Investigate any warning in the first hour before closing the change ticket.
