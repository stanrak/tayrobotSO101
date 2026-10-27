# Project State & Architecture Ledger

## 1. Project Overview
- **Project Name:** [Your Project Name]
- **Core Stack:** [e.g., React, Node.js, Express, PostgreSQL, Tailwind CSS]
- **Current Milestone:** [e.g., Sprint 1 - Authentication & RBAC]
- **Last Updated By:** [Claude / Gemini / Codex] - YYYY-MM-DD HH:MM

---

## 2. Active Roadmap & Task Checklist

### In Progress
- [ ] Implement JWT refresh token rotation mechanism (`/src/auth/token.ts`) - Assigned to: **Codex**

### Backlog / Next Up
- [ ] Security audit on authentication endpoints - Assigned to: **Gemini**
- [ ] Design RBAC schema migration (`/src/db/migrations/`) - Assigned to: **Claude**
- [ ] Unit tests for auth controller (coverage > 80%) - Assigned to: **Codex**

### Completed
- [x] Scaffold initial project structure and dependencies - [Claude]
- [x] Configure base environment variables and Docker setup - [Codex]

---

## 3. Architecture & Technical Decisions (ADR)
| Decision ID | Decision Summary | Rationale / Trade-offs | Status | Decided By |
|-------------|------------------|------------------------|--------|------------|
| ADR-001     | Use PostgreSQL + Prisma ORM | Strong schema validation and type safety | Accepted | Claude |
| ADR-002     | Argon2id for password hashing | Superior resistance to GPU cracking over bcrypt | Accepted | Gemini |

---

## 4. Known Blockers & Technical Debt
- **Blocker:** Missing OAuth client credentials for third-party provider (awaiting config).
- **Tech Debt:** Hardcoded CORS origins in dev mode; need environment-based abstraction.