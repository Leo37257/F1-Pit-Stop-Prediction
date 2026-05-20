/* pp-ui.jsx — shared design tokens + primitives for the pit-call prototype. */

const PP_T = {
  bg: "#0a0c10",
  bg2: "#0e1218",
  panel: "#11161d",
  panelHi: "#171d26",
  panelHover: "#1c232d",
  border: "#1d242e",
  borderHi: "#2a323d",
  borderBright: "#3a4453",
  ink: "#e6edf3",
  ink2: "#aeb7c2",
  ink3: "#8b95a3",
  muted: "#6b7480",
  mutedLow: "#525c69",
  red: "#ff4d4d",
  redDim: "#8b2a2a",
  redGlow: "rgba(255,77,77,0.18)",
  yellow: "#f5a524",
  green: "#3fb950",
  blue: "#58a6ff",
  cyan: "#39d0d8",
  violet: "#a371f7",
};

const PP_UI_FONT = `"Inter", "SF Pro Text", -apple-system, system-ui, sans-serif`;
const PP_MONO = `"JetBrains Mono", "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace`;

/* ----------------------------------------------------------------------------
   Compounds — Pirelli colors, name + short label.
---------------------------------------------------------------------------- */
const PP_COMPOUND = {
  S:        { color: "#ff2d2d", name: "SOFT",  full: "SOFT" },
  M:        { color: "#ffd84d", name: "MED",   full: "MEDIUM" },
  H:        { color: "#f5f5f5", name: "HARD",  full: "HARD" },
  I:        { color: "#2ecc71", name: "INTER", full: "INTER" },
  W:        { color: "#3a8dde", name: "WET",   full: "WET" },
  // Tolerate the raw CSV strings
  SOFT:     { color: "#ff2d2d", name: "SOFT",  full: "SOFT" },
  MEDIUM:   { color: "#ffd84d", name: "MED",   full: "MEDIUM" },
  HARD:     { color: "#f5f5f5", name: "HARD",  full: "HARD" },
  INTERMEDIATE: { color: "#2ecc71", name: "INTER", full: "INTER" },
  WET:      { color: "#3a8dde", name: "WET",   full: "WET" },
};

function PP_compound(c) {
  return PP_COMPOUND[c] || { color: PP_T.muted, name: "—", full: "—" };
}

/* ----------------------------------------------------------------------------
   Tire — Pirelli-style colored ring with a dark core.
---------------------------------------------------------------------------- */
function PPTire({ c = "M", label, full = false, size = 12, color }) {
  const cm = PP_compound(c);
  const inner = Math.max(3, size - 6);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, lineHeight: 1 }}>
      <span
        style={{
          position: "relative",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: size,
          height: size,
          borderRadius: "50%",
          background: cm.color,
          boxShadow: `0 0 0 1px rgba(0,0,0,0.5), 0 0 6px ${cm.color}55`,
          flexShrink: 0,
        }}
      >
        <span
          style={{
            width: inner,
            height: inner,
            borderRadius: "50%",
            background: "#0a0c10",
            boxShadow: `inset 0 0 0 1px ${cm.color}`,
          }}
        />
      </span>
      {label && (
        <span
          style={{
            fontFamily: PP_MONO,
            fontSize: 10.5,
            letterSpacing: 0.6,
            color: color || PP_T.ink2,
            fontWeight: 500,
          }}
        >
          {full ? cm.full : cm.name}
        </span>
      )}
    </span>
  );
}

/* ----------------------------------------------------------------------------
   Panel — bordered section with sticky head.
---------------------------------------------------------------------------- */
function PPPanel({ children, style, title, right, dense, scroll, noPad }) {
  return (
    <div
      style={{
        background: PP_T.panel,
        border: `1px solid ${PP_T.border}`,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        position: "relative",
        ...style,
      }}
    >
      {(title || right) && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "8px 12px",
            borderBottom: `1px solid ${PP_T.border}`,
            background: PP_T.bg2,
            flexShrink: 0,
            minHeight: 36,
          }}
        >
          <div
            style={{
              fontFamily: PP_UI_FONT,
              fontSize: 10.5,
              letterSpacing: 1.2,
              textTransform: "uppercase",
              color: PP_T.ink3,
              fontWeight: 600,
            }}
          >
            {title}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: PP_MONO, fontSize: 11, color: PP_T.muted }}>
            {right}
          </div>
        </div>
      )}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          padding: noPad ? 0 : dense ? 10 : 14,
          overflow: scroll ? "auto" : "hidden",
        }}
      >
        {children}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------------------
   Stat — KPI block (label · value · sub).
---------------------------------------------------------------------------- */
function PPStat({ label, value, sub, color, big, isMono = true, align = "left" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3, textAlign: align, minWidth: 0 }}>
      <div
        style={{
          fontFamily: PP_UI_FONT,
          fontSize: 9.5,
          letterSpacing: 1,
          textTransform: "uppercase",
          color: PP_T.muted,
          fontWeight: 600,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: isMono ? PP_MONO : PP_UI_FONT,
          fontSize: big ? 28 : 18,
          fontWeight: 500,
          color: color || PP_T.ink,
          letterSpacing: -0.2,
          lineHeight: 1.05,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontFamily: PP_MONO, fontSize: 10.5, color: PP_T.muted }}>
          {sub}
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------------------
   Tag / Dot / Btn
---------------------------------------------------------------------------- */
function PPTag({ children, color, bg, border, style }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        height: 20,
        padding: "0 8px",
        background: bg || "transparent",
        border: `1px solid ${border || PP_T.border}`,
        color: color || PP_T.ink2,
        fontFamily: PP_MONO,
        fontSize: 10.5,
        letterSpacing: 0.3,
        whiteSpace: "nowrap",
        ...style,
      }}
    >
      {children}
    </span>
  );
}

function PPDot({ color, size = 6, glow }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "50%",
        background: color,
        boxShadow: glow ? `0 0 8px ${color}` : "none",
        flexShrink: 0,
      }}
    />
  );
}

function PPBtn({ children, primary, danger, ghost, active, size = "md", icon, onClick, title, disabled, style }) {
  const heights = { sm: 22, md: 28, lg: 36 };
  const bgMap = active ? PP_T.ink : primary ? PP_T.ink : danger ? PP_T.red : ghost ? "transparent" : PP_T.panelHi;
  const fgMap = active ? PP_T.bg : primary ? PP_T.bg : danger ? "#fff" : PP_T.ink;
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      title={title}
      disabled={disabled}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        height: heights[size],
        padding: "0 10px",
        background: bgMap,
        color: fgMap,
        border: `1px solid ${active ? PP_T.ink : ghost ? PP_T.border : primary ? PP_T.ink : danger ? PP_T.red : PP_T.borderHi}`,
        fontFamily: PP_UI_FONT,
        fontSize: size === "lg" ? 13 : 11.5,
        fontWeight: 500,
        letterSpacing: 0.2,
        cursor: disabled ? "not-allowed" : "pointer",
        userSelect: "none",
        opacity: disabled ? 0.4 : 1,
        transition: "background 120ms ease, border-color 120ms ease, color 120ms ease",
        ...style,
      }}
      onMouseEnter={(e) => {
        if (disabled || active || primary || danger) return;
        e.currentTarget.style.background = PP_T.panelHover;
        e.currentTarget.style.borderColor = PP_T.borderBright;
      }}
      onMouseLeave={(e) => {
        if (disabled || active || primary || danger) return;
        e.currentTarget.style.background = ghost ? "transparent" : PP_T.panelHi;
        e.currentTarget.style.borderColor = ghost ? PP_T.border : PP_T.borderHi;
      }}
    >
      {icon && <span style={{ fontFamily: PP_MONO, opacity: 0.7 }}>{icon}</span>}
      {children}
    </button>
  );
}

/* ----------------------------------------------------------------------------
   Gauge — horizontal probability bar with optional threshold tick.
---------------------------------------------------------------------------- */
function PPGauge({ value, color = PP_T.red, h = 6, w = "100%", threshold }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div style={{ position: "relative", height: h, width: w, background: PP_T.bg2, border: `1px solid ${PP_T.border}` }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          width: `${v * 100}%`,
          background: color,
          transition: "width 320ms cubic-bezier(.4,.7,.3,1)",
        }}
      />
      {threshold !== undefined && (
        <div
          style={{
            position: "absolute",
            left: `${threshold * 100}%`,
            top: -3,
            bottom: -3,
            width: 2,
            background: PP_T.ink2,
          }}
        />
      )}
    </div>
  );
}

/* ----------------------------------------------------------------------------
   Sparkline / line chart
---------------------------------------------------------------------------- */
function PPSpark({ data, w = 160, h = 32, color = PP_T.blue, fill = false, threshold, markIndex }) {
  if (!data || data.length < 2) return <svg width={w} height={h} />;
  const xs = w / (data.length - 1);
  const min = Math.min(...data);
  const max = Math.max(...data);
  const norm = (v) => h - 2 - ((v - min) / (max - min || 1)) * (h - 4);
  const d = data.map((v, i) => `${i === 0 ? "M" : "L"} ${i * xs} ${norm(v)}`).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      {threshold !== undefined && (
        <line
          x1="0"
          x2={w}
          y1={norm(threshold)}
          y2={norm(threshold)}
          stroke={PP_T.borderHi}
          strokeDasharray="2 2"
          strokeWidth="1"
        />
      )}
      {fill && <path d={`${d} L ${w} ${h} L 0 ${h} Z`} fill={color} opacity="0.14" />}
      <path d={d} stroke={color} strokeWidth="1.4" fill="none" />
      {markIndex !== undefined && markIndex >= 0 && markIndex < data.length && (
        <circle
          cx={markIndex * xs}
          cy={norm(data[markIndex])}
          r="3"
          fill={PP_T.ink}
          stroke={color}
          strokeWidth="1.5"
        />
      )}
    </svg>
  );
}

function PPHRule({ vertical, color, style }) {
  return vertical ? (
    <div style={{ width: 1, alignSelf: "stretch", background: color || PP_T.border, ...style }} />
  ) : (
    <div style={{ height: 1, background: color || PP_T.border, ...style }} />
  );
}

Object.assign(window, {
  PP_T,
  PP_UI_FONT,
  PP_MONO,
  PP_COMPOUND,
  PP_compound,
  PPTire,
  PPPanel,
  PPStat,
  PPTag,
  PPDot,
  PPBtn,
  PPGauge,
  PPSpark,
  PPHRule,
});
