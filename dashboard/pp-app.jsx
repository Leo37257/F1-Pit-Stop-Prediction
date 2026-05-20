/* pp-app.jsx — F1 Pit-Stop Predictor · hi-fi prototype (A+B merged).
   Two-mode dashboard: Engineer Console (single driver focus) +
   Strategy Wall (multi-driver swim lanes). Shared scrub, race & driver state. */

/* global React, ReactDOM, PPData, PP_T, PP_UI_FONT, PP_MONO, PP_compound,
   PPTire, PPPanel, PPStat, PPTag, PPDot, PPBtn, PPGauge, PPSpark, PPHRule,
   TweaksPanel, TweakSection, TweakSlider, TweakToggle, TweakRadio, TweakSelect, useTweaks */

const { useState, useEffect, useMemo, useRef } = React;

/* ============================================================================
   Constants & helpers
============================================================================ */

const fmt = {
  lapTime(s) {
    if (!isFinite(s)) return "—";
    const m = Math.floor(s / 60);
    const sec = (s - m * 60).toFixed(3);
    return `${m}:${sec.padStart(6, "0")}`;
  },
  signed(n, dp = 2) {
    if (!isFinite(n)) return "—";
    const s = n.toFixed(dp);
    return n >= 0 ? "+" + s : s;
  },
  prob(p) {
    return p.toFixed(3);
  },
};

function attribution(row, baseRow) {
  // Toy SHAP-ish: rank features by deviation from a baseline.
  if (!row) return [];
  const cm = PP_compound(row.compound);
  // Magnitude scale so that no single feature monopolizes the bar at ±0.50;
  // most rows should show a clear hierarchy with 2-3 dominant features.
  // The model is sparse — when p(pit) is low, contributions should be small.
  const scale = Math.max(0.15, row.pPit * 0.7 + 0.1);
  const features = [
    {
      key: "TyreLife",
      value: row.tyreLife.toFixed(0),
      contrib: ((row.tyreLife - 8) / 30) * scale,
    },
    {
      key: "LapTime_Delta",
      value: fmt.signed(row.lapDelta, 2),
      contrib: (row.lapDelta / 3) * scale,
    },
    {
      key: "Cumulative_Degradation",
      value: row.cumDeg.toFixed(1),
      contrib: (-row.cumDeg / 18) * scale,
    },
    {
      key: `Compound · ${cm.full}`,
      value: row.compound,
      contrib: (row.compound === "S" ? 0.06 : row.compound === "M" ? 0.01 : -0.04) * scale * 2,
    },
    {
      key: "RaceProgress",
      value: row.raceProg.toFixed(2),
      contrib: (-0.12 + row.raceProg * 0.18) * scale,
    },
    {
      key: "Position",
      value: `P${row.position}`,
      contrib: ((10 - row.position) * 0.008) * scale,
    },
    {
      key: "Stint",
      value: row.stint.toString(),
      contrib: ((row.stint - 1) * 0.04) * scale,
    },
  ];
  return features
    .map((f) => ({ ...f, contrib: Math.max(-0.45, Math.min(0.45, f.contrib)) }))
    .sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib))
    .slice(0, 6);
}

/* ============================================================================
   AppBar — top chrome shared across modes
============================================================================ */

function AppBar({ raceData, mode, setMode, threshold, onOpenSettings, useRealModel, setUseRealModel, loadingReal }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 16px",
        height: 48,
        borderBottom: `1px solid ${PP_T.border}`,
        background: PP_T.panel,
        gap: 16,
        flexShrink: 0,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {/* Logo */}
        <div
          style={{
            width: 22,
            height: 22,
            background: PP_T.ink,
            color: PP_T.bg,
            fontFamily: PP_MONO,
            fontSize: 12,
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            letterSpacing: -0.5,
          }}
        >
          P
        </div>
        <div style={{ fontFamily: PP_UI_FONT, fontSize: 13, fontWeight: 600, letterSpacing: 0.2 }}>
          pitcall<span style={{ color: PP_T.muted }}>.dashboard</span>
        </div>
        <div style={{ fontFamily: PP_MONO, fontSize: 11.5, color: PP_T.muted }}>
          <span style={{ color: PP_T.border, margin: "0 10px" }}>/</span>
          {raceData.name.toLowerCase().replace(/\s+/g, "_")} · {raceData.year}
        </div>
      </div>

      {/* Mode tabs */}
      <div
        style={{
          display: "flex",
          border: `1px solid ${PP_T.borderHi}`,
          background: PP_T.bg2,
          height: 28,
        }}
      >
        {[
          { id: "console", label: "console", icon: "◧" },
          { id: "wall", label: "strategy wall", icon: "▦" },
        ].map((m) => {
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => setMode(m.id)}
              style={{
                border: "none",
                background: active ? PP_T.ink : "transparent",
                color: active ? PP_T.bg : PP_T.ink2,
                padding: "0 14px",
                fontFamily: PP_UI_FONT,
                fontSize: 11.5,
                fontWeight: active ? 600 : 500,
                letterSpacing: 0.3,
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                textTransform: "uppercase",
              }}
            >
              <span style={{ fontFamily: PP_MONO, opacity: 0.7 }}>{m.icon}</span>
              {m.label}
            </button>
          );
        })}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <PPTag border={PP_T.green} color={PP_T.green}>
          <PPDot color={PP_T.green} glow />
          REPLAY
        </PPTag>
        <PPTag>xgboost v2</PPTag>
        <PPTag color={PP_T.ink2}>F1 0.785</PPTag>
        <PPTag color={PP_T.ink2}>τ {threshold.toFixed(2)}</PPTag>
        <button
          type="button"
          onClick={() => setUseRealModel((v) => !v)}
          title="Toggle live inference from the FastAPI backend (replaces synthetic pPit for the focus driver)"
          style={{
            border: `1px solid ${useRealModel ? PP_T.green : PP_T.borderHi}`,
            background: useRealModel ? "rgba(0,255,136,0.08)" : "transparent",
            color: useRealModel ? PP_T.green : PP_T.muted,
            padding: "0 10px",
            height: 24,
            fontFamily: PP_MONO,
            fontSize: 10.5,
            fontWeight: 600,
            cursor: "pointer",
            letterSpacing: 0.5,
            textTransform: "uppercase",
          }}
        >
          {loadingReal ? "loading…" : (useRealModel ? "● real model" : "○ real model")}
        </button>
        <a
          href="/predict.html"
          style={{
            border: `1px solid ${PP_T.borderHi}`,
            background: "transparent",
            color: PP_T.ink2,
            padding: "0 10px",
            height: 24,
            display: "inline-flex",
            alignItems: "center",
            fontFamily: PP_MONO,
            fontSize: 10.5,
            fontWeight: 600,
            textDecoration: "none",
            letterSpacing: 0.5,
            textTransform: "uppercase",
          }}
        >
          ⇪ batch csv
        </a>
        <PPBtn ghost icon="⚙" onClick={onOpenSettings}>settings</PPBtn>
      </div>
    </div>
  );
}

/* ============================================================================
   Telemetry strip (Console mode)
============================================================================ */

function TelemetryStrip({ row, race }) {
  const cm = row ? PP_compound(row.compound) : null;
  const stats = [
    { l: "lap", v: row ? `${row.lap}` : "—", s: `of ${race.laps}` },
    { l: "stint", v: row ? `${row.stint}` : "—", s: row ? (row.stint > 1 ? `${row.stint - 1} stops` : "no stops yet") : "" },
    { l: "tyre", v: cm ? `${cm.name} · ${row.tyreLife}` : "—", s: "laps", c: cm?.color, badge: row && <PPTire c={row.compound} size={11} /> },
    { l: "position", v: row ? `P${row.position}` : "—", s: row ? `Δ ${row.posChange >= 0 ? "+" : ""}${row.posChange}` : "" },
    { l: "lap time", v: row ? fmt.lapTime(row.lapTime) : "—", s: row ? fmt.signed(row.lapDelta, 2) + "s" : "", c: row && row.lapDelta > 0.5 ? PP_T.red : undefined },
    { l: "deg.", v: row ? row.cumDeg.toFixed(1) : "—", s: "cumulative" },
    { l: "race prog.", v: row ? row.raceProg.toFixed(3) : "—", s: "RaceProgress" },
    { l: "p(pit)", v: row ? fmt.prob(row.pPit) : "—", s: row ? (row.pred === 1 ? "→ PIT" : "→ STAY") : "", c: row && row.pred === 1 ? PP_T.red : PP_T.green },
  ];
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(8, 1fr)",
        borderBottom: `1px solid ${PP_T.border}`,
        background: PP_T.panel,
        flexShrink: 0,
      }}
    >
      {stats.map((s, i) => (
        <div
          key={i}
          style={{
            padding: "12px 16px",
            borderRight: i < 7 ? `1px solid ${PP_T.border}` : "none",
            display: "flex",
            flexDirection: "column",
            gap: 3,
          }}
        >
          <div
            style={{
              fontFamily: PP_UI_FONT,
              fontSize: 9.5,
              letterSpacing: 1,
              textTransform: "uppercase",
              color: PP_T.muted,
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            {s.l}
            {s.badge}
          </div>
          <div
            style={{
              fontFamily: PP_MONO,
              fontSize: 18,
              fontWeight: 500,
              color: s.c || PP_T.ink,
              letterSpacing: -0.2,
            }}
          >
            {s.v}
          </div>
          {s.s && <div style={{ fontFamily: PP_MONO, fontSize: 10.5, color: PP_T.muted }}>{s.s}</div>}
        </div>
      ))}
    </div>
  );
}

/* ============================================================================
   Console mode — single-driver focus
============================================================================ */

function CallPanel({ row, threshold }) {
  if (!row) return <PPPanel title="next-lap call"><div /></PPPanel>;
  const isPit = row.pPit >= threshold;
  const correct = (row.pred === row.actual);
  const actualPit = row.actual === 1;
  return (
    <PPPanel
      title="next-lap call"
      right={
        <>
          <PPDot color={isPit ? PP_T.red : PP_T.green} glow />
          <span style={{ color: isPit ? PP_T.red : PP_T.green }}>
            {row.pPit > 0.75 ? "HIGH" : row.pPit > 0.4 ? "MED" : "LOW"} CONFIDENCE
          </span>
        </>
      }
      noPad
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", height: "100%" }}>
        <div
          style={{
            padding: "22px 26px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            borderRight: `1px solid ${PP_T.border}`,
          }}
        >
          <div style={{ fontFamily: PP_MONO, fontSize: 10, letterSpacing: 1.5, color: PP_T.muted, textTransform: "uppercase" }}>
            decision · lap {row.lap + 1}
          </div>
          <div
            style={{
              fontFamily: PP_UI_FONT,
              fontSize: 96,
              fontWeight: 800,
              letterSpacing: -3,
              color: isPit ? PP_T.red : PP_T.ink,
              lineHeight: 0.9,
              marginTop: 6,
              transition: "color 200ms ease",
            }}
          >
            {isPit ? "PIT" : "STAY"}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 14 }}>
            <PPDot color={correct ? PP_T.green : PP_T.red} />
            <span style={{ fontFamily: PP_MONO, fontSize: 11.5, color: correct ? PP_T.green : PP_T.red }}>
              actual = {actualPit ? "PIT" : "STAY"} · {correct ? "agreement ✓" : "disagreement ✗"}
            </span>
          </div>
        </div>
        <div
          style={{
            padding: "22px 26px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            gap: 16,
          }}
        >
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
            <span style={{ fontFamily: PP_UI_FONT, fontSize: 10, letterSpacing: 1.5, color: PP_T.muted, textTransform: "uppercase", fontWeight: 600 }}>
              P(pit) · class probability
            </span>
            <span style={{ fontFamily: PP_MONO, fontSize: 11, color: PP_T.muted }}>
              threshold {threshold.toFixed(2)}
            </span>
          </div>
          <div
            style={{
              fontFamily: PP_MONO,
              fontSize: 52,
              fontWeight: 500,
              color: PP_T.ink,
              letterSpacing: -2,
              transition: "color 200ms ease",
            }}
          >
            {fmt.prob(row.pPit)}
          </div>
          <PPGauge value={row.pPit} color={isPit ? PP_T.red : PP_T.green} h={8} threshold={threshold} />
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: PP_MONO, fontSize: 10, color: PP_T.muted }}>
            <span>0.00</span>
            <span style={{ color: PP_T.ink3 }}>τ = {threshold.toFixed(2)}</span>
            <span>1.00</span>
          </div>
          <div style={{ display: "flex", gap: 18, marginTop: 4, fontFamily: PP_MONO, fontSize: 11 }}>
            <span style={{ color: PP_T.muted }}>logit</span>
            <span style={{ color: PP_T.ink2 }}>{Math.log(row.pPit / (1 - row.pPit)).toFixed(2)}</span>
            <span style={{ color: PP_T.muted }}>·</span>
            <span style={{ color: PP_T.muted }}>margin</span>
            <span style={{ color: PP_T.ink2 }}>{fmt.signed(row.pPit - threshold, 2)}</span>
          </div>
        </div>
      </div>
    </PPPanel>
  );
}

function AttributionPanel({ row }) {
  const features = useMemo(() => attribution(row), [row]);
  return (
    <PPPanel
      title="feature attribution"
      right={<><span>method:</span><span style={{ color: PP_T.ink2 }}>shap (tree, approx)</span></>}
    >
      <div style={{ display: "grid", gap: 8 }}>
        {features.map((f) => {
          const sign = f.contrib >= 0 ? "+" : "-";
          const mag = Math.abs(f.contrib);
          const color = f.contrib >= 0 ? PP_T.red : PP_T.blue;
          return (
            <div key={f.key} style={{ display: "grid", gridTemplateColumns: "190px 1fr 60px 60px", gap: 10, alignItems: "center" }}>
              <span style={{ fontFamily: PP_MONO, fontSize: 11.5, color: PP_T.ink2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {f.key}
              </span>
              <div style={{ position: "relative", height: 12, background: PP_T.bg2, border: `1px solid ${PP_T.border}` }}>
                <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: PP_T.borderHi }} />
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    bottom: 0,
                    left: sign === "+" ? "50%" : `${50 - mag * 50}%`,
                    width: `${mag * 50}%`,
                    background: color,
                    opacity: 0.9,
                    transition: "left 220ms ease, width 220ms ease",
                  }}
                />
              </div>
              <span style={{ fontFamily: PP_MONO, fontSize: 11, color, textAlign: "right" }}>
                {sign}{mag.toFixed(2)}
              </span>
              <span style={{ fontFamily: PP_MONO, fontSize: 11, color: PP_T.muted, textAlign: "right" }}>
                {f.value}
              </span>
            </div>
          );
        })}
      </div>
    </PPPanel>
  );
}

function LapTrendPanel({ raceData, driver, currentLap }) {
  const history = PPData.getDriver(raceData, driver).filter((r) => r.lap <= currentLap);
  const tail = history.slice(-14);
  const data = tail.map((r) => r.lapTime);
  const last = tail[tail.length - 1];
  const pb = history.length ? Math.min(...history.map((r) => r.lapTime)) : 0;
  const avgLast5 = history.slice(-5).reduce((s, r) => s + r.lapTime, 0) / Math.max(1, Math.min(5, history.length));
  return (
    <PPPanel title={`lap-time trend · ${driver}`} right={<span style={{ color: PP_T.ink2 }}>last {tail.length} laps</span>}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <PPStat label="last lap" value={last ? fmt.lapTime(last.lapTime) : "—"} sub={last ? fmt.signed(last.lapTime - pb, 2) + "s vs PB" : ""} color={PP_T.red} />
        <PPSpark data={data} w={230} h={50} color={PP_T.red} fill markIndex={data.length - 1} />
      </div>
      <PPHRule />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginTop: 12 }}>
        <PPStat label="personal best" value={fmt.lapTime(pb)} sub="this stint" />
        <PPStat label="avg last 5" value={fmt.lapTime(avgLast5)} sub={avgLast5 > pb + 0.5 ? "degrading" : "stable"} color={avgLast5 > pb + 0.5 ? PP_T.yellow : PP_T.green} />
        <PPStat label="stint laps" value={last ? `${last.tyreLife}` : "—"} sub="on current set" />
      </div>
    </PPPanel>
  );
}

function FieldPanel({ raceData, currentLap, focusDriver, setFocusDriver, threshold, resolveRow }) {
  const rawRows = PPData.getLap(raceData, currentLap);
  const rows = resolveRow ? rawRows.map(resolveRow) : rawRows;
  const pitCalls = rows.filter((r) => r.pPit >= threshold).length;
  return (
    <PPPanel
      title="field · this lap"
      right={<span>{rows.length} cars · {pitCalls} pit calls</span>}
      noPad
      scroll
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "26px 60px 96px 30px 56px 56px 56px",
          padding: "8px 14px",
          borderBottom: `1px solid ${PP_T.border}`,
          fontFamily: PP_UI_FONT,
          fontSize: 9.5,
          letterSpacing: 1,
          textTransform: "uppercase",
          color: PP_T.muted,
          fontWeight: 600,
          background: PP_T.bg2,
          position: "sticky",
          top: 0,
          zIndex: 1,
        }}
      >
        <span>p</span><span>driver</span><span>tyre</span><span>age</span><span>Δ lap</span><span>p(pit)</span><span>call</span>
      </div>
      {rows.map((r, i) => {
        const isFocus = r.driver === focusDriver;
        const pit = r.pPit >= threshold;
        return (
          <button
            key={r.driver}
            type="button"
            onClick={() => setFocusDriver(r.driver)}
            style={{
              all: "unset",
              cursor: "pointer",
              display: "grid",
              gridTemplateColumns: "26px 60px 96px 30px 56px 56px 56px",
              padding: "8px 14px",
              borderBottom: `1px solid ${PP_T.border}`,
              fontFamily: PP_MONO,
              fontSize: 11.5,
              color: PP_T.ink2,
              background: isFocus ? "rgba(255,77,77,0.07)" : "transparent",
              borderLeft: isFocus ? `2px solid ${PP_T.red}` : "2px solid transparent",
              transition: "background 120ms ease",
            }}
            onMouseEnter={(e) => { if (!isFocus) e.currentTarget.style.background = PP_T.panelHi; }}
            onMouseLeave={(e) => { if (!isFocus) e.currentTarget.style.background = "transparent"; }}
          >
            <span style={{ color: PP_T.muted }}>{r.position}</span>
            <span style={{ color: isFocus ? PP_T.red : PP_T.ink, fontWeight: 500 }}>{r.driver}</span>
            <span><PPTire c={r.compound} label /></span>
            <span>{r.tyreLife}</span>
            <span style={{ color: PP_T.muted }}>{fmt.signed(r.lapDelta, 2)}</span>
            <span style={{ color: r.pPit >= threshold ? PP_T.red : PP_T.ink2 }}>{r.pPit.toFixed(2)}</span>
            <span style={{ color: pit ? PP_T.red : PP_T.muted, fontWeight: 600 }}>{pit ? "PIT" : "STAY"}</span>
          </button>
        );
      })}
    </PPPanel>
  );
}

function ConsoleMode({ raceData, currentLap, focusDriver, setFocusDriver, threshold, resolveRow }) {
  const rawFocusRow = PPData.getRow(raceData, currentLap, focusDriver);
  const focusRow = resolveRow ? resolveRow(rawFocusRow) : rawFocusRow;
  return (
    <>
      <TelemetryStrip row={focusRow} race={raceData} />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.25fr 1fr",
          gap: 1,
          background: PP_T.border,
          flex: 1,
          minHeight: 0,
        }}
      >
        {/* LEFT col */}
        <div style={{ display: "grid", gridTemplateRows: "minmax(220px, 1fr) 1fr", gap: 1, background: PP_T.border, minHeight: 0 }}>
          <CallPanel row={focusRow} threshold={threshold} />
          <AttributionPanel row={focusRow} />
        </div>
        {/* RIGHT col */}
        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", gap: 1, background: PP_T.border, minHeight: 0 }}>
          <LapTrendPanel raceData={raceData} driver={focusDriver} currentLap={currentLap} />
          <FieldPanel
            raceData={raceData}
            currentLap={currentLap}
            focusDriver={focusDriver}
            setFocusDriver={setFocusDriver}
            threshold={threshold}
            resolveRow={resolveRow}
          />
        </div>
      </div>
    </>
  );
}

/* ============================================================================
   Wall mode — multi-driver swim lanes
============================================================================ */

function KPIRow({ raceData, threshold }) {
  const summary = useMemo(() => PPData.summarize(raceData), [raceData]);
  const kpis = [
    { l: "pit calls (pred)", v: summary.predPit, s: "this race", c: PP_T.red },
    { l: "pit calls (actual)", v: summary.actualPit, s: "completed", c: PP_T.green },
    { l: "agreement", v: (summary.accuracy * 100).toFixed(1) + "%", s: `${summary.agree} / ${summary.total} rows` },
    { l: "macro F1", v: summary.f1.toFixed(3), s: "baseline 0.612" },
    { l: "precision · recall", v: `${summary.precision.toFixed(2)} · ${summary.recall.toFixed(2)}`, s: "current threshold" },
    { l: "decision τ", v: threshold.toFixed(2), s: "tunable", c: PP_T.cyan },
  ];
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(6, 1fr)",
        borderBottom: `1px solid ${PP_T.border}`,
        background: PP_T.panel,
        flexShrink: 0,
      }}
    >
      {kpis.map((k, i) => (
        <div key={i} style={{ padding: "14px 18px", borderRight: i < 5 ? `1px solid ${PP_T.border}` : "none" }}>
          <PPStat label={k.l} value={k.v} sub={k.s} color={k.c} big />
        </div>
      ))}
    </div>
  );
}

function SwimLanes({ raceData, currentLap, setCurrentLap, focusDriver, setFocusDriver, threshold }) {
  const drivers = raceData.drivers;
  const laps = raceData.laps;
  return (
    <PPPanel
      title="lap × driver · pit prediction"
      right={
        <>
          <PPDot color={PP_T.red} /><span>pred pit</span>
          <PPDot color={PP_T.green} style={{ marginLeft: 8 }} /><span>actual pit</span>
          <PPDot color={PP_T.yellow} style={{ marginLeft: 8 }} /><span>both (TP)</span>
          <span style={{ marginLeft: 8 }}>· click cell to inspect</span>
        </>
      }
      noPad
      scroll
    >
      {/* lap axis */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "160px 1fr",
          padding: "8px 14px",
          borderBottom: `1px solid ${PP_T.border}`,
          background: PP_T.bg2,
          position: "sticky",
          top: 0,
          zIndex: 2,
        }}
      >
        <span style={{ fontFamily: PP_UI_FONT, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase", color: PP_T.muted, fontWeight: 600 }}>
          driver / lap
        </span>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${laps}, 1fr)`, fontFamily: PP_MONO, fontSize: 9, color: PP_T.muted }}>
          {Array.from({ length: laps }, (_, i) => (
            <span key={i} style={{ textAlign: "center" }}>{(i + 1) % 5 === 0 ? i + 1 : ""}</span>
          ))}
        </div>
      </div>

      {drivers.map((d, di) => {
        const driverRows = PPData.getDriver(raceData, d);
        const isFocus = d === focusDriver;
        // Determine current compound for the label tire
        const currentRow = driverRows.find((r) => r.lap === currentLap) || driverRows[0];
        return (
          <div
            key={d}
            style={{
              display: "grid",
              gridTemplateColumns: "160px 1fr",
              alignItems: "center",
              padding: "5px 14px",
              borderBottom: `1px solid ${PP_T.border}`,
              background: isFocus ? "rgba(255,77,77,0.05)" : "transparent",
            }}
          >
            <button
              type="button"
              onClick={() => setFocusDriver(d)}
              style={{
                all: "unset",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 10,
              }}
            >
              <span style={{ fontFamily: PP_MONO, fontSize: 10, color: PP_T.muted, width: 22 }}>
                P{currentRow.position}
              </span>
              <span style={{ fontFamily: PP_MONO, fontSize: 12, color: isFocus ? PP_T.red : PP_T.ink, fontWeight: 500, width: 50 }}>
                {d}
              </span>
              <PPTire c={currentRow.compound} label />
            </button>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${laps}, 1fr)`, gap: 1, height: 20, position: "relative" }}>
              {driverRows.map((r) => {
                const pred = r.pPit >= threshold;
                const actual = r.actual === 1;
                let bg = PP_T.panelHi;
                let border = "none";
                if (pred && actual) bg = PP_T.yellow;
                else if (pred) bg = PP_T.red;
                else if (actual) bg = PP_T.green;
                else bg = `rgba(255,255,255,${0.02 + (r.pPit / 1.5)})`;

                const isActive = r.lap === currentLap;
                if (isActive) border = `1px solid ${PP_T.ink}`;
                return (
                  <button
                    key={r.lap}
                    type="button"
                    title={`L${r.lap} · ${r.compound} · age ${r.tyreLife} · p=${r.pPit.toFixed(2)}${actual ? " · actual PIT" : ""}`}
                    onClick={() => { setCurrentLap(r.lap); setFocusDriver(d); }}
                    style={{
                      all: "unset",
                      cursor: "pointer",
                      background: bg,
                      outline: border,
                      outlineOffset: -1,
                      transition: "background 120ms ease",
                    }}
                  />
                );
              })}
              {/* current lap vertical */}
              <div
                style={{
                  position: "absolute",
                  top: -2,
                  bottom: -2,
                  left: `${((currentLap - 1) / laps) * 100}%`,
                  width: `${100 / laps}%`,
                  border: `1px solid ${PP_T.ink}`,
                  pointerEvents: "none",
                }}
              />
            </div>
          </div>
        );
      })}
    </PPPanel>
  );
}

function SelectedDetail({ raceData, currentLap, focusDriver, threshold }) {
  const row = PPData.getRow(raceData, currentLap, focusDriver);
  if (!row) return null;
  const pit = row.pPit >= threshold;
  const correct = (row.pPit >= threshold ? 1 : 0) === row.actual;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "auto auto 1fr auto",
        gap: 22,
        padding: "12px 18px",
        borderTop: `1px solid ${PP_T.border}`,
        background: PP_T.panel,
        alignItems: "center",
        flexShrink: 0,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <span style={{ fontFamily: PP_UI_FONT, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase", color: PP_T.muted, fontWeight: 600 }}>
          selected
        </span>
        <span style={{ fontFamily: PP_MONO, fontSize: 12, color: PP_T.ink }}>{focusDriver} · L{currentLap}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ fontFamily: PP_UI_FONT, fontSize: 24, fontWeight: 700, color: pit ? PP_T.red : PP_T.ink, letterSpacing: -0.5 }}>
          {pit ? "PIT" : "STAY"}
        </span>
        <PPTag color={pit ? PP_T.red : PP_T.ink2} border={pit ? PP_T.redDim : PP_T.borderHi}>
          p = {row.pPit.toFixed(2)}
        </PPTag>
      </div>
      <div style={{ display: "flex", gap: 22, fontFamily: PP_MONO, fontSize: 11.5, color: PP_T.ink2 }}>
        <span><span style={{ color: PP_T.muted }}>tyre </span><PPTire c={row.compound} label /> · {row.tyreLife}</span>
        <span><span style={{ color: PP_T.muted }}>Δlap </span>{fmt.signed(row.lapDelta, 2)}s</span>
        <span><span style={{ color: PP_T.muted }}>deg </span>{row.cumDeg.toFixed(1)}</span>
        <span><span style={{ color: PP_T.muted }}>pos </span>P{row.position}</span>
        <span style={{ color: correct ? PP_T.green : PP_T.red }}>
          <PPDot color={correct ? PP_T.green : PP_T.red} /> actual: {row.actual ? "PIT" : "STAY"} {correct ? "✓" : "✗"}
        </span>
      </div>
    </div>
  );
}

function WallMode({ raceData, currentLap, setCurrentLap, focusDriver, setFocusDriver, threshold }) {
  return (
    <>
      <KPIRow raceData={raceData} threshold={threshold} />
      <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
        <SwimLanes
          raceData={raceData}
          currentLap={currentLap}
          setCurrentLap={setCurrentLap}
          focusDriver={focusDriver}
          setFocusDriver={setFocusDriver}
          threshold={threshold}
        />
      </div>
      <SelectedDetail raceData={raceData} currentLap={currentLap} focusDriver={focusDriver} threshold={threshold} />
    </>
  );
}

/* ============================================================================
   Timeline scrubber footer
============================================================================ */

function TimelineFooter({ raceData, currentLap, setCurrentLap, playing, setPlaying, focusDriver, threshold }) {
  const laps = raceData.laps;
  const focusRows = PPData.getDriver(raceData, focusDriver);
  const trackRef = useRef(null);

  const onScrub = (e) => {
    const rect = trackRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const lap = Math.max(1, Math.min(laps, Math.round(x * laps + 0.5)));
    setCurrentLap(lap);
  };
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    if (!dragging) return;
    const move = (e) => onScrub(e);
    const up = () => setDragging(false);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, [dragging]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "auto 1fr auto",
        gap: 16,
        padding: "10px 16px",
        borderTop: `1px solid ${PP_T.border}`,
        background: PP_T.panel,
        alignItems: "center",
        flexShrink: 0,
      }}
    >
      <div style={{ display: "flex", gap: 6 }}>
        <PPBtn icon="⏮" ghost onClick={() => { setPlaying(false); setCurrentLap(1); }} title="reset">reset</PPBtn>
        <PPBtn icon="◀" ghost onClick={() => { setPlaying(false); setCurrentLap(Math.max(1, currentLap - 1)); }} title="back" disabled={currentLap <= 1}>back</PPBtn>
        <PPBtn primary onClick={() => setPlaying((p) => !p)} title={playing ? "pause" : "play"}>
          {playing ? "❚❚ pause" : "▶ play"}
        </PPBtn>
        <PPBtn icon="▶|" ghost onClick={() => { setPlaying(false); setCurrentLap(Math.min(laps, currentLap + 1)); }} title="step" disabled={currentLap >= laps}>step</PPBtn>
      </div>

      <div style={{ position: "relative" }}>
        {/* lap ticks */}
        <div
          ref={trackRef}
          onMouseDown={(e) => { setDragging(true); onScrub(e); }}
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${laps}, 1fr)`,
            gap: 2,
            height: 24,
            cursor: "pointer",
            userSelect: "none",
          }}
        >
          {Array.from({ length: laps }, (_, i) => {
            const lap = i + 1;
            const focus = focusRows.find((r) => r.lap === lap);
            const pit = focus && focus.pPit >= threshold;
            const actual = focus && focus.actual === 1;
            const active = lap === currentLap;
            let bg = PP_T.panelHi;
            if (active) bg = PP_T.ink;
            else if (pit && actual) bg = PP_T.yellow;
            else if (pit) bg = PP_T.red;
            else if (actual) bg = PP_T.green;
            else if (focus) bg = `rgba(174,183,194,${0.08 + focus.pPit * 0.3})`;
            return <div key={lap} style={{ background: bg }} />;
          })}
        </div>
        {/* lap labels */}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontFamily: PP_MONO, fontSize: 9.5, color: PP_T.muted }}>
          <span>L1</span>
          <span>L{Math.round(laps / 4)}</span>
          <span>L{Math.round(laps / 2)}</span>
          <span>L{Math.round((laps * 3) / 4)}</span>
          <span>L{laps}</span>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 16, fontFamily: PP_MONO, fontSize: 11.5, color: PP_T.muted }}>
        <span>lap <span style={{ color: PP_T.ink, fontWeight: 600 }}>{currentLap}</span>/{laps}</span>
        <span>·</span>
        <span style={{ color: PP_T.ink2 }}>focus: {focusDriver}</span>
      </div>
    </div>
  );
}

/* ============================================================================
   Tweaks
============================================================================ */

const DEFAULT_TWEAKS = /*EDITMODE-BEGIN*/{
  "raceId": "singapore_2025",
  "threshold": 0.5,
  "showActualOverlay": true,
  "playSpeed": 600,
  "startMode": "console"
}/*EDITMODE-END*/;

function Tweaks({ tweaks, setTweak, focusDriver, setFocusDriver, raceData }) {
  const hasUploaded = PPData.hasUploaded && PPData.hasUploaded();
  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Race">
        <TweakSelect
          label="Session"
          value={tweaks.raceId}
          onChange={(v) => setTweak("raceId", v)}
          options={PPData.RACES.map((r) => ({
            value: r.id,
            label: r.id === "uploaded"
              ? `📤 ${r.name} · ${r.year} (uploaded)`
              : `${r.name} · ${r.year}`,
          }))}
        />
        <TweakSelect
          label="Focus driver"
          value={focusDriver}
          onChange={(v) => setFocusDriver(v)}
          options={raceData.drivers}
        />
        {hasUploaded && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 4 }}>
            <button
              type="button"
              onClick={() => {
                if (!confirm("Remove uploaded race and revert to demo data?")) return;
                PPData.clearUploaded();
                setTweak("raceId", "singapore_2025");
                // Force a hard reload so React picks up the mutated RACES list.
                window.location.assign("/");
              }}
              style={{
                border: `1px solid ${PP_T.borderHi}`,
                background: "transparent",
                color: PP_T.muted,
                padding: "4px 10px",
                fontFamily: PP_MONO,
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: 0.5,
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              ✕ clear uploaded · revert to demo
            </button>
          </div>
        )}
      </TweakSection>
      <TweakSection label="Model">
        <TweakSlider
          label="Decision threshold τ"
          value={tweaks.threshold}
          onChange={(v) => setTweak("threshold", v)}
          min={0.05}
          max={0.95}
          step={0.01}
        />
      </TweakSection>
      <TweakSection label="Playback">
        <TweakSlider
          label="Lap step (ms)"
          value={tweaks.playSpeed}
          onChange={(v) => setTweak("playSpeed", v)}
          min={120}
          max={1500}
          step={20}
          unit="ms"
        />
        <TweakRadio
          label="Default mode"
          value={tweaks.startMode}
          onChange={(v) => setTweak("startMode", v)}
          options={[
            { value: "console", label: "Console" },
            { value: "wall", label: "Wall" },
          ]}
        />
      </TweakSection>
    </TweaksPanel>
  );
}

/* ============================================================================
   App
============================================================================ */

function App() {
  const [tweaks, setTweak] = useTweaks(DEFAULT_TWEAKS);

  // If we landed via /?race=uploaded (from the CSV upload page) and an
  // uploaded race is actually present, switch to it. Run once on mount.
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const want = params.get("race");
      if (want && PPData.RACES_DATA[want] && want !== tweaks.raceId) {
        setTweak("raceId", want);
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const raceData = PPData.RACES_DATA[tweaks.raceId] || PPData.RACES_DATA.singapore_2025;
  const [mode, setMode] = useState(tweaks.startMode || "console");
  const [currentLap, setCurrentLap] = useState(Math.round(raceData.laps * 0.35));
  const [focusDriver, setFocusDriver] = useState(raceData.focus || raceData.drivers[0]);
  const [playing, setPlaying] = useState(false);
  const [useRealModel, setUseRealModel] = useState(false);
  const [realPreds, setRealPreds] = useState({}); // {[driver]: {[lap]: {pPit, pred}}}
  const [loadingReal, setLoadingReal] = useState(false);

  // When race changes, reset
  useEffect(() => {
    setCurrentLap(Math.round(raceData.laps * 0.35));
    setFocusDriver(raceData.focus || raceData.drivers[0]);
    setPlaying(false);
    setRealPreds({});
  }, [tweaks.raceId]);

  // Fetch real-model predictions for focusDriver when REAL MODEL is on
  useEffect(() => {
    if (!useRealModel) return;
    if (realPreds[focusDriver]) return; // cached
    const driverRows = PPData.getDriver(raceData, focusDriver);
    setLoadingReal(true);
    fetch("/predict/dashboard", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        race: raceData.name,
        year: raceData.year,
        rows: driverRows.map((r) => ({
          driver: r.driver,
          lap: r.lap,
          compound: r.compound,
          stint: r.stint,
          tyreLife: r.tyreLife,
          lapTime: r.lapTime,
          lapDelta: r.lapDelta,
          cumDeg: r.cumDeg,
          raceProg: r.raceProg,
          position: r.position,
          posChange: r.posChange,
          isStintStart: r.isStintStart ?? false,
        })),
      }),
    })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => {
        const byLap = {};
        data.predictions.forEach(({ lap, pPit, pred }) => { byLap[lap] = { pPit, pred }; });
        setRealPreds((prev) => ({ ...prev, [focusDriver]: byLap }));
      })
      .catch((err) => {
        console.error("Real-model fetch failed:", err);
        setUseRealModel(false);
        alert(`Real-model inference failed: ${err.message}. Falling back to synthetic.`);
      })
      .finally(() => setLoadingReal(false));
  }, [useRealModel, focusDriver, tweaks.raceId]);

  function resolveRow(row) {
    if (!row || !useRealModel) return row;
    const override = realPreds[row.driver]?.[row.lap];
    return override ? { ...row, ...override } : row;
  }

  // Playback timer
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setCurrentLap((l) => {
        if (l >= raceData.laps) {
          setPlaying(false);
          return raceData.laps;
        }
        return l + 1;
      });
    }, tweaks.playSpeed);
    return () => clearInterval(t);
  }, [playing, tweaks.playSpeed, raceData.laps]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      else if (e.key === "ArrowLeft") { setPlaying(false); setCurrentLap((l) => Math.max(1, l - 1)); }
      else if (e.key === "ArrowRight") { setPlaying(false); setCurrentLap((l) => Math.min(raceData.laps, l + 1)); }
      else if (e.key === "1") setMode("console");
      else if (e.key === "2") setMode("wall");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [raceData.laps]);

  const threshold = tweaks.threshold;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: `
          linear-gradient(0deg, rgba(255,255,255,0.012) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px),
          ${PP_T.bg}
        `,
        backgroundSize: "32px 32px, 32px 32px, auto",
        color: PP_T.ink,
        fontFamily: PP_UI_FONT,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <AppBar raceData={raceData} mode={mode} setMode={setMode} threshold={threshold} useRealModel={useRealModel} setUseRealModel={setUseRealModel} loadingReal={loadingReal} />
      {mode === "console" ? (
        <ConsoleMode
          raceData={raceData}
          currentLap={currentLap}
          focusDriver={focusDriver}
          setFocusDriver={setFocusDriver}
          threshold={threshold}
          resolveRow={resolveRow}
        />
      ) : (
        <WallMode
          raceData={raceData}
          currentLap={currentLap}
          setCurrentLap={setCurrentLap}
          focusDriver={focusDriver}
          setFocusDriver={setFocusDriver}
          threshold={threshold}
        />
      )}
      <TimelineFooter
        raceData={raceData}
        currentLap={currentLap}
        setCurrentLap={setCurrentLap}
        playing={playing}
        setPlaying={setPlaying}
        focusDriver={focusDriver}
        threshold={threshold}
      />
      <Tweaks tweaks={tweaks} setTweak={setTweak} focusDriver={focusDriver} setFocusDriver={setFocusDriver} raceData={raceData} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
