/* pp-data.jsx — synthesized race playback data.
   Exposes window.PPData with 3 races, each ~20 drivers × 50–60 laps.
   Deterministic via mulberry32 PRNG keyed off race.seed. */

(function () {
  function rng(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Generic driver pool — original IDs, no team branding.
  const DRIVERS = [
    { id: "D044", short: "044" },
    { id: "D016", short: "016" },
    { id: "D055", short: "055" },
    { id: "D004", short: "004" },
    { id: "FIS",  short: "FIS" },
    { id: "D011", short: "011" },
    { id: "BEA",  short: "BEA" },
    { id: "D023", short: "023" },
    { id: "D119", short: "119" },
    { id: "D063", short: "063" },
    { id: "D091", short: "091" },
    { id: "D102", short: "102" },
    { id: "D018", short: "018" },
    { id: "D007", short: "007" },
    { id: "D032", short: "032" },
    { id: "D041", short: "041" },
    { id: "D024", short: "024" },
    { id: "D017", short: "017" },
    { id: "D077", short: "077" },
    { id: "D003", short: "003" },
  ];

  // Each race has its own tyre limits and base lap time.
  const RACES = [
    { id: "singapore_2025", name: "Singapore Grand Prix", year: 2025, laps: 62, seed: 7,  baseTime: 100.2, focus: "FIS",  compoundBias: ["M","M","M","M","M","S","S","M","H","M","M","S","M","H","M","S","M","H","M","M"] },
    { id: "british_2024",   name: "British Grand Prix",   year: 2024, laps: 52, seed: 11, baseTime:  93.2, focus: "D119", compoundBias: ["M","H","M","M","S","S","H","M","H","S","M","H","S","M","M","H","M","M","H","S"] },
    { id: "spa_2023",       name: "Belgian Grand Prix",   year: 2023, laps: 44, seed: 5,  baseTime: 107.6, focus: "D044", compoundBias: ["M","M","H","M","H","M","M","M","M","H","M","S","M","H","M","M","H","M","M","S"] },
  ];

  function maxLifeFor(c) { return c === "S" ? 18 : c === "M" ? 28 : 38; }
  function degRateFor(c) { return c === "S" ? 0.11 : c === "M" ? 0.07 : 0.045; }

  function generateRace(race) {
    const r = rng(race.seed);
    const drivers = DRIVERS.slice(0, 20);
    const laps = race.laps;

    // Plan stints per driver — first stop ~1/3 race in, optional second stop.
    const stints = drivers.map((d, di) => {
      const startCompound = race.compoundBias[di] || "M";
      // first stop: bias by start compound life
      const baseStop = startCompound === "S" ? 14 : startCompound === "M" ? 22 : 30;
      const firstStop = Math.max(8, Math.min(laps - 6, baseStop + Math.floor((r() - 0.5) * 8)));
      const secondCompound = startCompound === "S" ? "M" : startCompound === "M" ? "H" : "M";
      const wantTwoStop = r() < 0.35;
      const secondStop = wantTwoStop
        ? Math.min(laps - 4, firstStop + 14 + Math.floor(r() * 8))
        : null;

      const plan = [{ start: 1, end: firstStop, compound: startCompound }];
      if (secondStop) {
        plan.push({ start: firstStop + 1, end: secondStop, compound: secondCompound });
        plan.push({ start: secondStop + 1, end: laps, compound: "M" });
      } else {
        plan.push({ start: firstStop + 1, end: laps, compound: secondCompound });
      }
      return plan;
    });

    // Build row per (driver, lap)
    const rows = [];
    // Track each driver's running position
    const startPos = drivers.map((_, i) => i + 1);

    for (let lap = 1; lap <= laps; lap++) {
      const lapRows = drivers.map((d, di) => {
        const driverStints = stints[di];
        const stintIdx = driverStints.findIndex((s) => lap >= s.start && lap <= s.end);
        const stint = driverStints[Math.max(0, stintIdx)];
        const tyreLife = lap - stint.start + 1;
        const compound = stint.compound;
        const isStintStart = stintIdx > 0 && lap === stint.start; // just pitted at end of last lap

        const ml = maxLifeFor(compound);
        const dr = degRateFor(compound);

        // Lap time
        const base = race.baseTime + (compound === "S" ? -0.5 : compound === "H" ? 0.7 : 0);
        const wear = tyreLife * dr;
        const noise = (r() - 0.5) * 0.45;
        const fuelEffect = -lap * 0.02; // car gets lighter
        const lapTime = base + wear + fuelEffect + noise + (isStintStart ? 4 : 0); // out-lap penalty

        // Stint-relative best
        const lapDelta = wear + noise * 0.6;
        const cumDeg = -tyreLife * 0.35 - (di % 5) * 0.4 - r() * 0.6;
        const raceProg = lap / laps;

        // P(pit) — sparse signal: ramp aggressively in the final 3 laps of a
        // stint and stay low elsewhere. Most laps a driver is NOT about to pit.
        const nextStintBreak = driverStints.find((s) => s.start === lap + 1);
        const stintEnd = stint.end;
        const lapsUntilPit = nextStintBreak ? 0 : (lap === stintEnd ? 0 : stintEnd - lap);

        let pPit;
        if (nextStintBreak) {
          pPit = 0.74 + r() * 0.22;                        // about to pit
        } else if (lapsUntilPit === 1) {
          pPit = 0.32 + r() * 0.22;                        // one lap out
        } else if (lapsUntilPit === 2) {
          pPit = 0.16 + r() * 0.14;
        } else if (lapsUntilPit === 3) {
          pPit = 0.08 + r() * 0.10;
        } else {
          // Baseline — slight tyre-age influence + noise. Capped low.
          pPit = 0.02 + Math.min(0.18, (tyreLife / ml) * 0.18) + (r() - 0.5) * 0.05;
        }
        // 4% chance of model getting confused (false high) — keeps Wall mode honest
        if (r() < 0.04 && !nextStintBreak) pPit = Math.max(pPit, 0.55 + r() * 0.15);
        // 8% chance of false low at actual pit (model miss)
        if (nextStintBreak && r() < 0.08) pPit = 0.30 + r() * 0.15;

        // Suppress just after pit
        if (tyreLife < 3) pPit = Math.min(pPit, 0.05);

        pPit = Math.max(0.01, Math.min(0.99, pPit));

        const pred = pPit >= 0.5 ? 1 : 0;
        const actual = nextStintBreak ? 1 : 0;

        return {
          lap,
          driver: d.id,
          short: d.short,
          compound,
          stint: Math.max(1, stintIdx + 1),
          tyreLife,
          lapTime,
          lapDelta,
          cumDeg,
          raceProg,
          posChange: 0,
          pPit,
          pred,
          actual,
          isStintStart,
        };
      });

      // Position: sort by lapTime cumulative-ish — keep mostly stable order with small perturbations
      // For simplicity, base position on starting grid + small swap rule
      const sorted = lapRows
        .map((row, i) => ({ row, score: startPos[i] + (row.lapDelta || 0) * 0.05 + (row.compound === "S" ? -0.2 : 0) }))
        .sort((a, b) => a.score - b.score);

      sorted.forEach((s, i) => {
        s.row.position = i + 1;
        s.row.posChange = startPos.indexOf(drivers.findIndex((dr) => dr.id === s.row.driver) + 1) === i ? 0 : 0;
      });

      // Update startPos very slightly based on this lap's order so positions evolve
      sorted.forEach((s, i) => {
        const di = drivers.findIndex((dr) => dr.id === s.row.driver);
        startPos[di] = startPos[di] * 0.85 + (i + 1) * 0.15;
      });

      rows.push(...lapRows);
    }

    return {
      ...race,
      drivers: drivers.map((d) => d.id),
      driverMeta: drivers,
      stints,
      rows,
    };
  }

  const RACES_DATA = {};
  RACES.forEach((race) => {
    RACES_DATA[race.id] = generateRace(race);
  });

  function getLap(raceData, lap) {
    return raceData.rows.filter((r) => r.lap === lap).sort((a, b) => a.position - b.position);
  }
  function getDriver(raceData, driver) {
    return raceData.rows.filter((r) => r.driver === driver).sort((a, b) => a.lap - b.lap);
  }
  function getRow(raceData, lap, driver) {
    return raceData.rows.find((r) => r.lap === lap && r.driver === driver);
  }
  // Race-level summary
  function summarize(raceData) {
    const total = raceData.rows.length;
    const predPit = raceData.rows.filter((r) => r.pred === 1).length;
    const actualPit = raceData.rows.filter((r) => r.actual === 1).length;
    const agree = raceData.rows.filter((r) => r.pred === r.actual).length;
    let tp = 0, fp = 0, fn = 0, tn = 0;
    raceData.rows.forEach((r) => {
      if (r.pred === 1 && r.actual === 1) tp++;
      else if (r.pred === 1 && r.actual === 0) fp++;
      else if (r.pred === 0 && r.actual === 1) fn++;
      else tn++;
    });
    const prec = tp / Math.max(1, tp + fp);
    const rec = tp / Math.max(1, tp + fn);
    const f1 = (2 * prec * rec) / Math.max(0.0001, prec + rec);
    return { total, predPit, actualPit, agree, accuracy: agree / total, tp, fp, fn, tn, precision: prec, recall: rec, f1 };
  }

  window.PPData = { RACES, RACES_DATA, getLap, getDriver, getRow, summarize };
})();
