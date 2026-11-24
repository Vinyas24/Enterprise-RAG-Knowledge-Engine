# API Reference

## Authentication

Service clients use the OAuth2 client credentials flow. Exchange the client id and secret for a bearer token at `/oauth/token`, then send `Authorization: Bearer <token>`. Rotate client secrets every 180 days. User-facing clients use short-lived tokens issued by the identity provider.

## Pagination

List endpoints use cursor pagination. Pass `limit` and `cursor`; the response contains `next_cursor` and `has_more`. The default page size is 20 and the maximum page size is 100. Cursors are opaque and expire after 24 hours.

## Rate limits

Each client may send 1000 requests per minute. When the quota is exceeded the API returns status code 429 with a `Retry-After` header. Retry with exponential backoff; do not retry non-idempotent requests without an idempotency key.

## Errors

Errors use one envelope with `code`, `message` and `request_id`. A 401 means the token is missing or expired, 403 means the caller lacks clearance, and 404 hides resources that belong to another tenant. Include the request id when contacting support.
