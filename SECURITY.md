# Security Policy

## Supported versions

openspective is pre-1.0. Security fixes are applied to the latest `main` and the most recent
release only.

| Version | Supported          |
|---------|--------------------|
| `main`  | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Use GitHub's [private vulnerability reporting](https://github.com/Ninhache/Openspective/security/advisories/new)
to report a vulnerability privately. We aim to acknowledge reports within 72 hours and to provide a
remediation timeline after triage.

When reporting, please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal request/response or PoC is ideal).
- Affected version / commit.

## Hardening notes for self-hosters

openspective is designed to run on infrastructure you control. For internet-facing deployments:

- Enable authentication with `OPENSPECTIVE_API_TOKENS` and serve over TLS (terminate at a reverse proxy).
- Enable rate limiting with `OPENSPECTIVE_RATE_LIMIT` to mitigate abuse.
- Keep `/metrics` and the operational endpoints off the public internet, or behind your proxy's auth.
- Do not expose Redis publicly; keep it on a private network.
