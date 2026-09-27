# Technical Risk Checklist

Guidance for spotting technical risks in BRD/SRD documents. Each entry: what to look for in the
requirements, why it is a risk, and the usual mitigation.

## Architecture: unquantified scalability
Look for: "must scale", "high volume", "millions of users", peak events (Black Friday, month-end, tax deadline)
without numbers for concurrent users, transactions per second, data growth or peak-to-average ratio.
Risk: architecture sized on guesses; performance failure at the first real peak.
Mitigation: agree measurable NFRs (peak TPS, concurrency, p95 latency), capacity model, load test at 1.5–2x peak before go-live.

## Architecture: single points of failure
Look for: one database, one message broker, one data centre/region, a single external dependency on the critical path,
batch jobs that must finish before business opens.
Risk: one component outage stops the whole service; availability target cannot be met.
Mitigation: redundancy per tier, multi-AZ deployment, graceful degradation, documented failover tested regularly.

## Availability: target without disaster recovery
Look for: 99.9% / 99.99% / 24x7 / "zero downtime" with no RTO, RPO, DR site, backup or maintenance-window requirements.
Risk: 99.99% allows ~52 minutes downtime per year; without DR design and operational maturity the SLA is unattainable.
Mitigation: define RTO/RPO, DR strategy (active-active vs warm standby), zero-downtime deployment approach, SLA exclusions.

## Performance: real-time expectations
Look for: "real-time", "instant", "immediately visible" without latency targets; synchronous chains across several systems.
Risk: ambiguous expectations; synchronous coupling to slow or legacy systems breaks latency and availability.
Mitigation: define latency SLOs, use asynchronous/event-driven updates where "near real-time" suffices, caching.

## Security: requirements missing or TBD
Look for: security section empty, "TBD", "standard security", no mention of authentication, authorisation, roles,
encryption, audit logging, secrets management, penetration testing.
Risk: security designed late causes rework, failed audits, delayed go-live, breach exposure.
Mitigation: threat model early, security NFRs (OWASP ASVS level), IAM/SSO design, pen test in plan.

## Security: personal and regulated data
Look for: PII (name, email, date of birth, address, phone), payment card data, health data, national IDs, children's data,
cross-border data transfer, data residency.
Risk: GDPR/CCPA/PCI-DSS/HIPAA non-compliance, fines, breach notification obligations.
Mitigation: data classification, minimisation, encryption at rest/in transit, retention and deletion rules, DPIA,
tokenisation for card data (reduce PCI scope), consent management.

## Integration: legacy systems
Look for: mainframe, AS400/iSeries, COBOL, "existing system", batch file interfaces, systems owned by other teams.
Risk: limited APIs, batch-only data, low throughput, scarce skills, slow change processes, test environment shortages.
Mitigation: early interface spike, integration layer/anti-corruption layer, agree SLAs and test environments with owning team.

## Integration: third-party services
Look for: payment gateways, SMS/email providers, identity providers, SaaS APIs, credit bureaus, mapping services.
Risk: outages, rate limits, contract/pricing changes, API version deprecation, vendor lock-in, sandbox differs from production.
Mitigation: timeouts, retries with backoff, circuit breakers, fallback behaviour, contract SLAs, abstraction layer.

## Data: migration from existing systems
Look for: "migrate", "existing balances/records/history", "cut-over", "data conversion".
Risk: poor source data quality, mapping gaps, reconciliation failures, long cut-over window, no rollback.
Mitigation: data profiling early, mapping specification, repeated trial migrations, automated reconciliation, rollback plan.

## Data: volumes, retention and reporting
Look for: long history retention, analytics/reporting on the transactional database, archiving not mentioned, large files.
Risk: database growth degrades performance; reporting load impacts transactions; storage cost.
Mitigation: retention policy, archiving, read replicas or separate analytics store.

## Technology: emerging or specialised technology
Look for: machine learning/AI, personalisation, blockchain, new frameworks, technology new to the organisation.
Risk: skills gap, immature tooling, data availability/quality for ML, unclear accuracy acceptance criteria, model governance.
Mitigation: proof of concept, measurable success criteria, MLOps/monitoring, training or specialist hiring.

## Requirements quality: ambiguity and TBDs
Look for: TBD, TBC, "to be decided", "as appropriate", "user friendly", "fast", "secure", requirements without acceptance criteria.
Risk: scope creep, rework, disputes at UAT, estimates unreliable.
Mitigation: resolve open items before design sign-off, add measurable acceptance criteria, requirements traceability.

## Requirements quality: BRD and SRD inconsistency
Look for: business requirement in the BRD with no corresponding system requirement in the SRD; different numbers
(users, volumes, SLAs, dates) between documents; SRD features not justified by any BRD requirement.
Risk: delivered system does not meet business needs; gold-plating; late discovery of missing scope.
Mitigation: traceability matrix BRD -> SRD -> test cases; reconcile conflicting figures with the business owner.

## Delivery: fixed date before peak season
Look for: go-live tied to a fixed business event (holiday season, regulatory deadline, fiscal year) with large scope.
Risk: compressed testing, no time to stabilise before peak load, change freeze conflicts.
Mitigation: phased release, feature toggles, go-live well before peak, hypercare, contingency plan.

## Operations: missing non-functional operations requirements
Look for: no monitoring, alerting, logging, support model, runbooks, environments (dev/test/staging), CI/CD requirements.
Risk: incidents detected by customers; slow recovery; unmanageable production.
Mitigation: observability requirements, SLOs and alerting, on-call and support model, environment strategy.
