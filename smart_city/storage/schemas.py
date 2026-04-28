# Created by Metrum AI for AMD

"""SQL DDL for all TimescaleDB tables and hypertables."""

DDL = """
CREATE TABLE IF NOT EXISTS users (
    user_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username  TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role      TEXT NOT NULL DEFAULT 'viewer',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active  BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS crowd_counts (
    timestamp     TIMESTAMPTZ NOT NULL,
    stream_id     INTEGER NOT NULL,
    zone_id       TEXT NOT NULL,
    zone_name     TEXT,
    person_count  INTEGER,
    density_score FLOAT,
    severity      TEXT,
    city          TEXT,
    lat           FLOAT,
    lon           FLOAT,
    fps           FLOAT
);
SELECT create_hypertable('crowd_counts', 'timestamp', if_not_exists => TRUE);
SELECT add_retention_policy('crowd_counts', INTERVAL '7 days',
       if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS crowd_alerts (
    alert_id           UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp          TIMESTAMPTZ NOT NULL,
    stream_id          INTEGER NOT NULL,
    zone_id            TEXT NOT NULL,
    zone_name          TEXT,
    zone_type          TEXT,
    violation_type     TEXT,
    location_name      TEXT,
    lat                FLOAT,
    lon                FLOAT,
    person_count       INTEGER,
    threshold          INTEGER,
    severity           TEXT,
    trend              TEXT,
    description        TEXT,
    is_auto_popup      BOOLEAN DEFAULT FALSE,
    status             TEXT DEFAULT 'active',
    acknowledged_by    TEXT,
    acknowledged_at    TIMESTAMPTZ,
    acknowledge_reason TEXT,
    PRIMARY KEY (alert_id, timestamp)
);
SELECT create_hypertable('crowd_alerts', 'timestamp', if_not_exists => TRUE);
SELECT add_retention_policy('crowd_alerts', INTERVAL '90 days',
       if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_alerts_zone_ts
    ON crowd_alerts (zone_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON crowd_alerts (severity, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status
    ON crowd_alerts (status, timestamp DESC);

CREATE TABLE IF NOT EXISTS llm_reports (
    report_id    UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp    TIMESTAMPTZ NOT NULL,
    zone_id      TEXT,
    report_type  TEXT,
    content      TEXT,
    operator_id  TEXT,
    PRIMARY KEY (report_id, timestamp)
);
SELECT create_hypertable('llm_reports', 'timestamp', if_not_exists => TRUE);
SELECT add_retention_policy('llm_reports', INTERVAL '365 days',
       if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS recurring_patterns (
    pattern_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id             TEXT NOT NULL,
    zone_name           TEXT,
    pattern_description TEXT,
    frequency           TEXT,
    peak_day            TEXT,
    peak_hour_start     INTEGER,
    peak_hour_end       INTEGER,
    avg_count           FLOAT,
    occurrences         INTEGER,
    last_updated        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (zone_id, peak_day, peak_hour_start)
);
CREATE INDEX IF NOT EXISTS idx_patterns_zone
    ON recurring_patterns (zone_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_recurring_patterns_zone_day_hour
    ON recurring_patterns (zone_id, peak_day, peak_hour_start);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id      UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    operator_id TEXT NOT NULL,
    action      TEXT NOT NULL,
    target_id   TEXT,
    details     JSONB,
    ip_address  TEXT,
    PRIMARY KEY (log_id, timestamp)
);
SELECT create_hypertable('audit_log', 'timestamp', if_not_exists => TRUE);
SELECT add_retention_policy('audit_log', INTERVAL '365 days',
       if_not_exists => TRUE);

-- Simulator: tag simulated rows so the API / dashboard can label them
ALTER TABLE crowd_counts
    ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN DEFAULT FALSE;
ALTER TABLE crowd_alerts
    ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN DEFAULT FALSE;

-- Simulator state – singleton row (state_id always = 1)
CREATE TABLE IF NOT EXISTS simulator_state (
    state_id              INTEGER PRIMARY KEY DEFAULT 1,
    mode                  TEXT NOT NULL DEFAULT 'active',
    seeded_days           INTEGER,
    seed_start            TIMESTAMPTZ,
    seed_end              TIMESTAMPTZ,
    real_data_from        TIMESTAMPTZ,
    retire_after_days     INTEGER NOT NULL DEFAULT 7,
    last_checked          TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_mode CHECK (
        mode IN ('active', 'retiring', 'retired')
    )
);
"""
