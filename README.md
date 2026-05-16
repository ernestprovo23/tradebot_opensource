# Auto Trade Bot — Open-Source Edition

> A reference architecture for a multi-vehicle algorithmic trading system, intentionally vague on alpha and configuration so other engineers can study the **shape** of the system and the **AI-driven workflow** used to build it.

---

## ⚠️ Important Disclaimers

**This is an educational reference, not a turnkey product.**

- All trading decisions and outcomes are yours alone
- You are responsible for API keys, secret management, risk parameters, testing, and regulatory compliance
- Default settings in this repo are **placeholders** and **must** be replaced before any live use
- The author and contributors accept no liability for losses, outages, or misconfigurations

> Paper trade extensively. Then paper trade again.

---

## What This Repo Is

This repository captures the **public, non-proprietary surface** of a multi-engine algorithmic trading platform built in Python. It is intended for engineers who want to see:

1. How a real multi-vehicle trading system is **structured** (engines, scheduler, risk, diagnostics)
2. How **AI-assisted development** (Claude Code as primary collaborator) was used to design, implement, test, debug, and operate it — roughly 99% of the work
3. What **defensive-first** risk architecture looks like in practice (tiered circuit breakers, writer-integrity guards, autonomous order management)

This repo deliberately **does not** publish:

- Signal weights, coefficient priors, regime playbooks, or thesis content
- Allocation percentages, position-sizing formulas, or stop-loss thresholds
- Symbol universes, broker account details, or P&L
- Server hostnames, internal IPs, MongoDB URIs, or any secrets

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       Scheduler  (APScheduler)                     │
│   heartbeat · account_health · portfolio_snapshot · order_cleanup  │
│   self_healing · system_diagnostics · pnl_attribution · per-engine │
└─────────────┬──────────────┬──────────────┬───────────────┬────────┘
              │              │              │               │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Crypto   │  │  Options  │  │ Equities  │  │Commodities│
        │  Engine   │  │  Engine   │  │  Engine   │  │  Engine   │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │               │
              └──────────────┴──────┬───────┴───────────────┘
                                    │
                ┌───────────────────▼───────────────────┐
                │             BaseEngine                │
                │  · Alpaca HTTP w/ bounded retry       │
                │  · order polling + poll_timeout guard │
                │  · writer-integrity P&L drop          │
                │  · position reconciliation            │
                └───────────────────┬───────────────────┘
                                    │
                ┌───────────────────▼───────────────────┐
                │            Risk Subsystem             │
                │  · per-vehicle circuit breakers       │
                │    (consecutive · daily · weekly      │
                │     · monthly · emergency)            │
                │  · order manager (stale cancel)       │
                │  · account guardian                   │
                │  · autonomous manager                 │
                └───────────────────┬───────────────────┘
                                    │
                ┌───────────────────▼───────────────────┐
                │     Intelligence / Diagnostics        │
                │  · SystemDiagnostician (LLM-assisted) │
                │  · P&L Attribution (LLM narratives)   │
                │  · Market regime detection            │
                └───────────────────────────────────────┘

External: Alpaca · MongoDB · AWS SES · LLM provider(s)
```

For a richer breakdown see **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

---

## Core Components (Public Surface Only)

### `RiskManagement`
Trade validation, position sizing, portfolio rebalancing, P&L-aware risk adjustment, and performance reporting. The *interface* is published here; the formulas you must supply.

### Options Analytics
Black-Scholes implied volatility and Greeks (Δ, Γ, Θ, ν). `greekscomp.py` holds the math. Usable as-is for educational purposes.

### Data Management
- **Broker**: Alpaca paper/live REST API (set keys via env vars only)
- **Storage**: MongoDB for analytics, snapshots, and historical state
- **Reporting**: AWS SES for daily/weekly briefings

### Communications
Optional Microsoft Teams / Slack webhook integration for monitoring.

---

## Design Philosophy — "Defensive Fortress"

> Never lose money. Build capital inch by inch.

This is not a maximize-return system; it is a *minimize-blowup* system. Concretely:

- **Tiered circuit breakers** halt or scale-down trading on consecutive losses, daily drawdown, weekly drawdown, monthly drawdown, and an emergency kill-switch — *per vehicle*, with global overrides
- **Writer-integrity guard**: any "trade" that didn't actually fill (poll_timeout, rejection, non-fill) has its P&L coerced to `null` before persistence — no phantom accounting
- **Authoritative reconciliation**: each cycle re-reads positions from the broker; local cache never overrides reality
- **Autonomous order management**: stale orders are cancelled automatically; pending fills are time-budgeted

---

## How This Was Built — Claude Code, ~99%

This is the part most reference repos won't show you. The development workflow that produced this system:

| Layer | Tool | Role |
|-------|------|------|
| **Orchestrator** | Claude Code (Opus 4.x) | Plans, decomposes, dispatches |
| **Specialist agents** | Subagents (security, DB, full-stack, DevOps, QA) | Domain-specific implementation |
| **Skills** | Versioned skill files | Repeatable workflows (health checks, deploys, reviews) |
| **Slash commands** | Custom `/trading-health`, `/deploy`, `/secops-review` | One-keystroke ops |
| **Verification** | `verification-before-completion` skill | No "done" claims without evidence |
| **Memory** | File-backed memory across sessions | Continuity without leakage |

**Practical patterns we relied on**:

- **Plan → approve → execute** loop for every non-trivial change (no inline freelancing)
- **Parallel agent dispatch** for independent work streams
- **Background monitors** (filesystem tails, log greps) for long-running stability checks
- **Pre-push security scans** as a non-skippable step before any public commit (this commit included)

Result: a senior engineer can supervise — not type — the implementation of a multi-engine production-class system.

---

## Required Setup

### 1. API Configuration (env-based, never hardcoded)

Create a `.env` (already in `.gitignore`):

```bash
ALPACA_API_KEY=YOUR_KEY
ALPACA_SECRET_KEY=YOUR_SECRET
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # paper by default
MONGO_DB_CONN_STRING=mongodb://localhost:27017/
# Optional
TEAMS_WEBHOOK_URL=
SLACK_WEBHOOK_URL=
```

### 2. Risk Parameters

Edit `risk_params.json` — the shipped values are deliberately placeholders.

### 3. Customizations You **Must** Supply

| Item | Why |
|------|-----|
| Position sizing formula | Strategy-specific; not published |
| Stop-loss / take-profit logic | Vehicle-specific |
| Sector / correlation caps | Risk tolerance varies |
| Symbol universes | Capacity and capital constraints |
| Signal generation | The *alpha*; you bring it |

---

## Quickstart (Paper Only)

```bash
# 1. Clone + install
git clone https://github.com/ernestprovo23/tradebot_opensource.git
cd tradebot_opensource
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # then edit with your keys (paper Alpaca!)
nano risk_params.json  # set real numbers

# 3. Smoke test
python3 quote.py       # confirm Alpaca connectivity
python3 account_monitoring.py   # confirm account read
```

---

## Repo Layout

```
analytics/         data movement utilities (env-driven now)
claudeprojects/    AI workflow artifacts
comms/             Teams / Slack / portfolio-history helpers
crypto/            crypto-vehicle entry points
options/           options-vehicle entry points + risk strategy
models/            scratch space + migration helpers
server_setup/      deployment notes
```

---

## Contributing

PRs welcome — especially around:
- Better risk frameworks (without injecting opinions on parameters)
- Test coverage
- Documentation clarity
- Additional broker abstractions

Please do **not** open PRs that hardcode credentials, IPs, account sizes, or proprietary strategy logic.

---

## License

MIT — see [LICENSE](./LICENSE).

---

## Critical Reminders

- ✅ Paper trade for weeks before any live capital
- ✅ Rotate every API key the moment it touches a public repo
- ✅ Use environment variables exclusively; never commit secrets
- ✅ Set monitoring + alerts before unattended operation
- ✅ Implement your own backup/recovery
- ❌ Do not assume any defaults in this repo are safe

If this repo helped you, the best thank-you is to publish your own architecture write-up so the next engineer learns faster. That's the loop.
