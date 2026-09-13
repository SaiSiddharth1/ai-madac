# AI MADAC — Implementation Plan

**Status key:** `[x]` verified in the current source code · `[ ]` still to complete · `[~]` implemented in part or requires end-to-end validation.

## 1. Product objective

AI MADAC is a web application where authenticated users upload CSV/XLSX datasets, ask natural-language questions, and receive multi-agent analysis, charts, reports, and optional machine-learning results.

## 2. Current architecture

```text
Next.js frontend
  -> FastAPI API
     -> PostgreSQL metadata and uploaded dataset tables
     -> LangGraph workflow
        Supervisor -> SQL / Data Analyst / ML / Visualization / RAG -> Report
     -> File storage for charts and trained models
```

## 3. Completed implementation

### Foundation and infrastructure

- [x] FastAPI application with a startup lifecycle and health endpoint.
- [x] PostgreSQL-backed SQLAlchemy data model and asynchronous database sessions.
- [x] Docker Compose configuration for PostgreSQL and the backend.
- [x] Environment configuration template for database, LLM, storage, and application settings.
- [x] Static file route for generated charts.

### Authentication

- [x] User registration with password hashing and JWT issuance.
- [x] User login with credential verification and JWT issuance.
- [x] Protected API dependencies for the current authenticated user.
- [x] Password-reset API using a time-limited OTP.
- [x] Frontend register, login, and password-reset screens.

### Dataset management

- [x] CSV and XLSX upload support.
- [x] File-size and empty-file validation.
- [x] CSV encoding recovery, conservative data cleaning, and normalized UTF-8 download.
- [x] Column-name sanitization and dataset schema extraction.
- [x] Uploaded data persisted into a dedicated database table.
- [x] Dataset metadata, listing, detail, profile, and deletion APIs.
- [x] Dataset dashboard, upload, list, and detail screens.
- [x] Dataset profiling is triggered after successful upload.

### Multi-agent analysis and reporting

- [x] LangGraph workflow with Supervisor, SQL, Data Analyst, ML, Visualization, RAG, and Report agents.
- [x] Supervisor-directed, conditional agent routing.
- [x] Standard natural-language query endpoint.
- [x] Streaming query endpoint with real-time agent-status events.
- [x] Chat interface that consumes streamed query events and displays query history.
- [x] Query, agent-run, and report persistence.
- [x] Report list API plus PDF and JSON exports.
- [x] Reports dashboard.
- [x] Automatic visual-analysis routing for data-oriented answers.
- [x] Browser-ready chart images shown alongside report summaries and tables.
- [x] Visual charts included in PDF report exports.

### Machine learning

- [x] Backend APIs for model training, listing trained models, and prediction.
- [x] Storage for trained-model metadata and files.
- [x] Frontend flow for model training, metrics review, and prediction input.

## 4. Remaining implementation checklist

### Priority 1 — Complete existing user flows

- [x] Connect dataset uploads to vector-store indexing so the RAG agent can retrieve uploaded dataset schema context.
- [x] Add cleanup of vector-store records when a dataset is deleted.
- [ ] Add re-indexing when an existing dataset is replaced or its schema changes.
- [~] Confirm the vector-store runtime dependency is installed and works in the deployment image.
- [x] Build the ML frontend: choose a dataset, select target/features, start training, show metrics, list models, and run predictions.
- [x] Replace the static Models page with live backend data.
- [ ] Verify the complete happy path: register -> upload -> profile -> streamed query -> chart/report -> export.

### Priority 2 — Reliability and quality

- [ ] Add backend unit tests for authentication, authorization, uploads, dataset deletion, SQL validation, agent routing, and report exports.
- [ ] Add frontend tests for authentication, upload, dataset browsing, streaming chat, and report downloads.
- [ ] Add end-to-end tests covering the primary user journey.
- [ ] Add migration management for database schema changes and a repeatable seed/sample-data workflow.
- [ ] Verify streaming-query cancellation, error reporting, and cleanup for long-running agent tasks.
- [ ] Record meaningful per-agent execution timing and failure details rather than only aggregate results.
- [ ] Add user-friendly retry and error states across the frontend.

### Priority 3 — Security and data protection

- [ ] Add rate limiting to login, registration, password-reset, upload, and query endpoints.
- [ ] Ensure production secrets are only supplied through secure environment configuration; do not retain sample/default credentials.
- [ ] Add password-reset abuse protections, such as request throttling and OTP attempt limits.
- [ ] Define retention and deletion rules for uploaded tables, charts, reports, trained models, and vector entries.
- [ ] Review uploaded-file handling and database permissions for production deployment.

### Priority 4 — Production deployment and observability

- [ ] Add and enable the frontend Docker service; it is currently commented out in Docker Compose.
- [ ] Create a production Docker configuration without development reload mode.
- [ ] Add deployment documentation, including first-run database setup and required environment variables.
- [ ] Add health/readiness checks for the database, LLM provider, and background dependencies.
- [ ] Add structured logs, error tracking, and monitoring for API, agents, uploads, model training, and report exports.
- [ ] Add database backup and recovery procedures.

## 5. Suggested delivery milestones

### Milestone A — End-to-end analysis MVP

- [ ] Validate a real user can authenticate, upload a supported dataset, receive a profile, ask a question, and export a report.
- [ ] Fix all failures found in this path.
- [ ] Add automated coverage for that path.

### Milestone B — Retrieval and machine learning

- [ ] Enable and test dataset-schema retrieval for the RAG agent.
- [ ] Complete the ML model management and prediction interface.
- [ ] Add validation and clear messaging for unsuitable training datasets.

### Milestone C — Production readiness

- [ ] Complete security hardening, deployment configuration, observability, backups, and operational documentation.
- [ ] Run load, security, and acceptance testing before release.

## 6. Definition of done

- [ ] All Priority 1 items are complete and covered by automated tests.
- [ ] Users can complete the main workflow without manual database intervention.
- [ ] Failures are visible to users and diagnosable through logs/monitoring.
- [ ] Deployment uses production-safe secrets, configuration, and backup procedures.
- [ ] The plan is updated whenever a milestone item is completed.
