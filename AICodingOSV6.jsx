import { useState, useRef, useCallback } from "react";

const STRATEGIES = [
  {
    id: "A",
    name: "Conservative",
    color: "#4ade80",
    desc: "Minimal change, preserve existing logic",
    systemPrompt: `You are a CONSERVATIVE code patch agent. Your goal: fix the issue with MINIMAL changes.
Rules:
- Change as few lines as possible
- Preserve all existing logic and patterns
- No refactoring, no style changes
- Safety first, correctness over elegance
Return a JSON object ONLY (no markdown, no explanation):
{
  "patch": "full corrected code here",
  "changes_summary": "brief description of what changed",
  "confidence": 0.0-1.0
}`
  },
  {
    id: "B",
    name: "Aggressive",
    color: "#f97316",
    desc: "Rewrite problematic section with modern approach",
    systemPrompt: `You are an AGGRESSIVE code patch agent. Your goal: fix the issue with a bold, modern rewrite.
Rules:
- Rewrite the problematic section completely
- Use modern patterns, better algorithms if possible
- Optimize for performance and correctness
- Don't fear breaking changes if they're improvements
Return a JSON object ONLY (no markdown, no explanation):
{
  "patch": "full corrected code here",
  "changes_summary": "brief description of what changed",
  "confidence": 0.0-1.0
}`
  },
  {
    id: "C",
    name: "Refactor",
    color: "#818cf8",
    desc: "Fix + improve structure and maintainability",
    systemPrompt: `You are a REFACTOR code patch agent. Your goal: fix the issue AND improve code structure.
Rules:
- Fix the bug/issue
- Improve naming, structure, separation of concerns
- Add comments where logic is non-obvious
- Optimize for long-term maintainability
Return a JSON object ONLY (no markdown, no explanation):
{
  "patch": "full corrected code here",
  "changes_summary": "brief description of what changed",
  "confidence": 0.0-1.0
}`
  }
];

const SCORER_PROMPT = `You are a code quality scoring engine. Evaluate a code patch against the original issue.
Score each dimension 0.0-1.0:
- tests_pass_rate: does it logically solve the problem?
- performance: is it efficient?
- complexity: is it simple/readable? (1.0 = simple)
- security: any obvious vulnerabilities? (1.0 = secure)
- maintainability: easy to maintain?
Return a JSON object ONLY (no markdown):
{
  "tests_pass_rate": 0.0-1.0,
  "performance": 0.0-1.0,
  "complexity": 0.0-1.0,
  "security": 0.0-1.0,
  "maintainability": 0.0-1.0,
  "total_score": 0.0-1.0,
  "verdict": "one sentence assessment",
  "tests_passed": number,
  "tests_total": number
}`;

const FAILURE_SIM_PROMPT = `You are a failure simulation engine. Analyze code for potential failure modes.
Check: large inputs, bad inputs, race conditions, memory pressure, security issues, edge cases.
Return JSON ONLY:
{
  "failure_modes": [
    {"scenario": "name", "risk": "low|medium|high", "impact": "description", "mitigation": "fix"}
  ],
  "overall_risk": "low|medium|high"
}`;

const DARWIN_PROMPT = `You are a Darwin mutation agent. Take the winning patch and make it slightly better.
Rules:
- Keep what works
- Fix remaining weaknesses
- Small targeted improvements only
- Don't break what's already good
Return JSON ONLY:
{
  "patch": "improved code here",
  "changes_summary": "what was mutated",
  "confidence": 0.0-1.0
}`;

const REPO_ANALYZER_PROMPT = `You are a repository graph analyzer. Analyze the provided code and extract structure.
Return JSON ONLY:
{
  "files": ["list of files/modules detected"],
  "main_components": ["key classes/functions"],
  "dependencies": ["external deps detected"],
  "test_files": ["test files if any"],
  "entry_points": ["main entry points"],
  "complexity": "low|medium|high",
  "language": "detected language"
}`;

async function callClaude(systemPrompt, userContent) {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: userContent }]
    })
  });
  const data = await resp.json();
  const text = data.content?.map(b => b.text || "").join("") || "";
  const clean = text.replace(/```json\n?|```/g, "").trim();
  return JSON.parse(clean);
}

const PHASES = [
  "idle", "analyzing_repo", "generating_patches",
  "scoring_patches", "selecting_winner", "darwin_evolving",
  "failure_simulation", "final_report"
];

const phaseLabel = {
  idle: "Ready",
  analyzing_repo: "Analyzing Repository...",
  generating_patches: "Generating Patch Candidates...",
  scoring_patches: "Scoring Patches...",
  selecting_winner: "Selecting Winner...",
  darwin_evolving: "Darwin Evolution...",
  failure_simulation: "Failure Simulation...",
  final_report: "Complete"
};

const riskColor = r => r === "high" ? "#f87171" : r === "medium" ? "#fbbf24" : "#4ade80";
const scoreColor = s => s >= 0.85 ? "#4ade80" : s >= 0.7 ? "#fbbf24" : "#f87171";

export default function AICodingOSV6() {
  const [phase, setPhase] = useState("idle");
  const [code, setCode] = useState("");
  const [task, setTask] = useState("");
  const [selectedStrategies, setSelectedStrategies] = useState(["A", "B", "C"]);
  const [repoGraph, setRepoGraph] = useState(null);
  const [patches, setPatches] = useState([]);
  const [scores, setScores] = useState([]);
  const [winner, setWinner] = useState(null);
  const [darwinGen, setDarwinGen] = useState(0);
  const [failureModes, setFailureModes] = useState(null);
  const [log, setLog] = useState([]);
  const [error, setError] = useState("");
  const [memory, setMemory] = useState(null);
  const logRef = useRef(null);

  const addLog = useCallback((msg, type = "info") => {
    setLog(prev => [...prev, { msg, type, ts: Date.now() }]);
    setTimeout(() => logRef.current?.scrollTo(0, logRef.current.scrollHeight), 50);
  }, []);

  const saveMemory = useCallback(async (data) => {
    try {
      await window.storage.set("aicos_memory", JSON.stringify(data));
    } catch {}
  }, []);

  const loadMemory = useCallback(async () => {
    try {
      const r = await window.storage.get("aicos_memory");
      if (r) setMemory(JSON.parse(r.value));
    } catch {}
  }, []);

  const toggleStrategy = (id) => {
    setSelectedStrategies(prev => {
      if (prev.includes(id)) {
        // Prevent deselecting the last strategy
        if (prev.length === 1) return prev;
        return prev.filter(s => s !== id);
      }
      return [...prev, id];
    });
  };

  const reset = () => {
    setPhase("idle"); setRepoGraph(null); setPatches([]);
    setScores([]); setWinner(null); setDarwinGen(0);
    setFailureModes(null); setLog([]); setError("");
  };

  const run = async () => {
    if (!code.trim() || !task.trim()) {
      setError("Paste your code and describe the task/bug.");
      return;
    }
    if (selectedStrategies.length === 0) {
      setError("Select at least one strategy.");
      return;
    }
    setError(""); reset();

    const activeStrategies = STRATEGIES.filter(s => selectedStrategies.includes(s.id));

    try {
      // PHASE 1: Repo Analysis
      setPhase("analyzing_repo");
      addLog("🔍 Building repo graph...", "system");
      const graph = await callClaude(REPO_ANALYZER_PROMPT, `Code:\n${code}`);
      setRepoGraph(graph);
      addLog(`✓ Detected: ${graph.language} | Complexity: ${graph.complexity}`, "success");
      addLog(`  Files: ${graph.files?.join(", ") || "N/A"}`, "info");

      // PHASE 2: Generate Patches
      setPhase("generating_patches");
      addLog(`⚡ Generating ${activeStrategies.length} patch candidate(s) in parallel...`, "system");
      const patchResults = await Promise.all(
        activeStrategies.map(async (s) => {
          addLog(`  → Strategy ${s.id}: ${s.name}`, "info");
          const result = await callClaude(s.systemPrompt,
            `Task/Bug: ${task}\n\nOriginal Code:\n${code}`);
          return { ...s, ...result };
        })
      );
      setPatches(patchResults);
      addLog(`✓ All ${activeStrategies.length} patch(es) generated`, "success");

      // PHASE 3: Score Patches
      setPhase("scoring_patches");
      addLog("📊 Scoring all patches...", "system");
      const scoreResults = await Promise.all(
        patchResults.map(async (p) => {
          const score = await callClaude(SCORER_PROMPT,
            `Original task: ${task}\nPatch ${p.id} (${p.name}):\n${p.patch}\nChanges: ${p.changes_summary}`);
          addLog(`  Patch ${p.id}: score=${score.total_score?.toFixed(2)} — ${score.verdict}`, "info");
          return { ...p, score };
        })
      );
      setScores(scoreResults);

      // PHASE 4: Select Winner
      setPhase("selecting_winner");
      const sorted = [...scoreResults].sort((a, b) => (b.score?.total_score || 0) - (a.score?.total_score || 0));
      let currentWinner = sorted[0];
      setWinner(currentWinner);
      addLog(`🏆 Winner: Patch ${currentWinner.id} (${currentWinner.name}) — ${currentWinner.score?.total_score?.toFixed(2)}`, "success");

      // PHASE 5: Darwin Evolution
      if ((currentWinner.score?.total_score || 0) < 0.85) {
        setPhase("darwin_evolving");
        addLog("🧬 Score < 0.85 — Starting Darwin evolution (max 3 generations)...", "system");
        let gen = 0;
        let evolving = true;
        let evolved = currentWinner;
        while (gen < 3 && evolving) {
          gen++;
          setDarwinGen(gen);
          addLog(`  Generation ${gen}: mutating winner...`, "info");
          const mutated = await callClaude(DARWIN_PROMPT,
            `Original task: ${task}\nCurrent patch (score ${evolved.score?.total_score?.toFixed(2)}):\n${evolved.patch}\nWeaknesses: ${evolved.score?.verdict}`);
          const newScore = await callClaude(SCORER_PROMPT,
            `Original task: ${task}\nMutated patch:\n${mutated.patch}\nChanges: ${mutated.changes_summary}`);
          addLog(`  Gen ${gen} score: ${newScore.total_score?.toFixed(2)}`, "info");
          if ((newScore.total_score || 0) > (evolved.score?.total_score || 0)) {
            evolved = { ...evolved, ...mutated, score: newScore };
            addLog(`  ✓ Improved to ${newScore.total_score?.toFixed(2)}`, "success");
          } else {
            addLog(`  ✗ No improvement, stopping evolution`, "warning");
            evolving = false;
          }
          if ((newScore.total_score || 0) >= 0.85) evolving = false;
        }
        setWinner(evolved);
        currentWinner = evolved;
        addLog(`🏁 Darwin complete: final score ${currentWinner.score?.total_score?.toFixed(2)}`, "success");
      } else {
        addLog("✓ Score ≥ 0.85 — skipping Darwin evolution", "info");
      }

      // PHASE 6: Failure Simulation
      setPhase("failure_simulation");
      addLog("💥 Running failure simulation...", "system");
      const failures = await callClaude(FAILURE_SIM_PROMPT,
        `Task: ${task}\nFinal patch:\n${currentWinner.patch}`);
      setFailureModes(failures);
      addLog(`✓ Risk: ${failures.overall_risk} | Modes found: ${failures.failure_modes?.length || 0}`, "success");

      // Save memory
      const memData = {
        repo_graph: graph,
        selected_strategies: selectedStrategies,
        patch_history: scoreResults.map(p => ({ id: p.id, score: p.score?.total_score })),
        winning_strategy: currentWinner.name,
        remaining_risks: failures.failure_modes?.filter(f => f.risk === "high").map(f => f.scenario) || []
      };
      await saveMemory(memData);
      setMemory(memData);

      setPhase("final_report");
      addLog("✅ V6 Pipeline complete!", "success");
    } catch (e) {
      setError("Pipeline error: " + e.message);
      addLog("❌ Error: " + e.message, "error");
      setPhase("idle");
    }
  };

  const isRunning = phase !== "idle" && phase !== "final_report";

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f", color: "#e2e8f0", fontFamily: "'JetBrains Mono', 'Fira Code', monospace", padding: "20px", boxSizing: "border-box" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "24px" }}>
        <div style={{ fontSize: "11px", color: "#6366f1", letterSpacing: "4px", marginBottom: "4px" }}>AI CODING OS</div>
        <div style={{ fontSize: "28px", fontWeight: 700, background: "linear-gradient(135deg, #6366f1, #818cf8, #4ade80)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>AGENT V6</div>
        <div style={{ fontSize: "11px", color: "#64748b", marginTop: "4px" }}>Darwin Patch Evolution Laboratory</div>
      </div>

      {/* Pipeline Phases */}
      <div style={{ display: "flex", gap: "4px", marginBottom: "20px", overflowX: "auto", padding: "4px 0" }}>
        {PHASES.filter(p => p !== "idle").map((p, i) => {
          const idx = PHASES.indexOf(phase);
          const pIdx = PHASES.indexOf(p);
          const active = phase === p;
          const done = idx > pIdx && phase !== "idle";
          return (
            <div key={p} style={{ flex: 1, minWidth: "80px", padding: "6px 4px", borderRadius: "4px", textAlign: "center", fontSize: "9px", border: `1px solid ${active ? "#6366f1" : done ? "#4ade8044" : "#1e293b"}`, background: active ? "#6366f120" : done ? "#4ade8010" : "transparent", color: active ? "#818cf8" : done ? "#4ade80" : "#475569", transition: "all 0.3s" }}>
              {done ? "✓" : active ? "◆" : `${i + 1}`}<br />{p.replace(/_/g, " ")}
            </div>
          );
        })}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
        {/* Input */}
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", padding: "16px" }}>
          <div style={{ fontSize: "10px", color: "#6366f1", letterSpacing: "2px", marginBottom: "10px" }}>INPUT</div>
          <textarea
            value={code}
            onChange={e => setCode(e.target.value)}
            placeholder="Paste your code here..."
            style={{ width: "100%", height: "160px", background: "#1e293b", border: "1px solid #334155", borderRadius: "4px", color: "#e2e8f0", padding: "10px", fontSize: "11px", resize: "vertical", boxSizing: "border-box", fontFamily: "monospace" }}
          />
          <input
            value={task}
            onChange={e => setTask(e.target.value)}
            placeholder="Describe the bug or task..."
            style={{ width: "100%", marginTop: "8px", background: "#1e293b", border: "1px solid #334155", borderRadius: "4px", color: "#e2e8f0", padding: "10px", fontSize: "12px", boxSizing: "border-box" }}
          />

          {/* Strategy Selection */}
          <div style={{ marginTop: "12px" }}>
            <div style={{ fontSize: "9px", color: "#475569", letterSpacing: "2px", marginBottom: "8px" }}>STRATEGIES</div>
            <div style={{ display: "flex", gap: "8px" }}>
              {STRATEGIES.map(s => {
                const isSelected = selectedStrategies.includes(s.id);
                const isLast = selectedStrategies.length === 1 && isSelected;
                return (
                  <button
                    key={s.id}
                    onClick={() => toggleStrategy(s.id)}
                    disabled={isRunning || isLast}
                    title={isLast ? "At least one strategy must be selected" : s.desc}
                    style={{
                      flex: 1,
                      padding: "8px 4px",
                      borderRadius: "6px",
                      border: `2px solid ${isSelected ? s.color : "#334155"}`,
                      background: isSelected ? `${s.color}15` : "#1e293b",
                      color: isSelected ? s.color : "#475569",
                      fontSize: "10px",
                      fontWeight: isSelected ? 700 : 400,
                      cursor: isRunning || isLast ? "not-allowed" : "pointer",
                      transition: "all 0.2s",
                      opacity: isLast ? 0.6 : 1,
                      fontFamily: "inherit",
                    }}
                  >
                    <div style={{ fontSize: "12px", marginBottom: "2px" }}>{s.id}</div>
                    <div>{s.name}</div>
                  </button>
                );
              })}
            </div>
            <div style={{ fontSize: "9px", color: "#334155", marginTop: "6px" }}>
              {selectedStrategies.length} of {STRATEGIES.length} strategies active
            </div>
          </div>

          {error && <div style={{ color: "#f87171", fontSize: "11px", marginTop: "8px" }}>{error}</div>}
          <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
            <button onClick={run} disabled={isRunning}
              style={{ flex: 1, padding: "10px", background: isRunning ? "#334155" : "linear-gradient(135deg, #6366f1, #4ade80)", border: "none", borderRadius: "6px", color: "#fff", fontSize: "12px", fontWeight: 700, cursor: isRunning ? "not-allowed" : "pointer", letterSpacing: "1px" }}>
              {isRunning ? "EVOLVING..." : "▶ RUN V6"}
            </button>
            <button onClick={reset} disabled={isRunning}
              style={{ padding: "10px 16px", background: "#1e293b", border: "1px solid #334155", borderRadius: "6px", color: "#94a3b8", fontSize: "11px", cursor: "pointer" }}>
              RESET
            </button>
          </div>
        </div>

        {/* Log */}
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", padding: "16px" }}>
          <div style={{ fontSize: "10px", color: "#6366f1", letterSpacing: "2px", marginBottom: "10px" }}>EXECUTION LOG</div>
          <div ref={logRef} style={{ height: "230px", overflowY: "auto", fontSize: "10px", lineHeight: "1.8" }}>
            {log.length === 0 && <div style={{ color: "#334155" }}>Waiting for input...</div>}
            {log.map((l, i) => (
              <div key={i} style={{ color: l.type === "success" ? "#4ade80" : l.type === "error" ? "#f87171" : l.type === "warning" ? "#fbbf24" : l.type === "system" ? "#818cf8" : "#94a3b8" }}>
                {l.msg}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Repo Graph */}
      {repoGraph && (
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "10px", color: "#6366f1", letterSpacing: "2px", marginBottom: "12px" }}>REPO GRAPH</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", fontSize: "11px" }}>
            {[
              { label: "Language", val: repoGraph.language },
              { label: "Complexity", val: repoGraph.complexity },
              { label: "Components", val: repoGraph.main_components?.join(", ") || "—" },
              { label: "Entry Points", val: repoGraph.entry_points?.join(", ") || "—" },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: "#1e293b", borderRadius: "4px", padding: "8px" }}>
                <div style={{ color: "#475569", fontSize: "9px", marginBottom: "4px" }}>{label}</div>
                <div style={{ color: "#e2e8f0" }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Patch Candidates */}
      {scores.length > 0 && (
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "10px", color: "#6366f1", letterSpacing: "2px", marginBottom: "12px" }}>PATCH CANDIDATES</div>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${scores.length}, 1fr)`, gap: "12px" }}>
            {scores.map(p => {
              const isWin = winner?.id === p.id;
              const s = p.score || {};
              return (
                <div key={p.id} style={{ border: `2px solid ${isWin ? p.color : "#1e293b"}`, borderRadius: "8px", padding: "14px", background: isWin ? `${p.color}08` : "#1e293b", position: "relative" }}>
                  {isWin && <div style={{ position: "absolute", top: "8px", right: "8px", fontSize: "10px", background: p.color, color: "#000", padding: "2px 8px", borderRadius: "10px", fontWeight: 700 }}>WINNER</div>}
                  <div style={{ color: p.color, fontWeight: 700, marginBottom: "4px" }}>Patch {p.id} — {p.name}</div>
                  <div style={{ color: "#64748b", fontSize: "10px", marginBottom: "10px" }}>{p.desc}</div>
                  <div style={{ fontSize: "24px", fontWeight: 700, color: scoreColor(s.total_score), marginBottom: "8px" }}>
                    {s.total_score?.toFixed(2) || "—"}
                  </div>
                  {["tests_pass_rate", "performance", "complexity", "security", "maintainability"].map(k => (
                    <div key={k} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px", fontSize: "9px" }}>
                      <div style={{ color: "#475569", width: "100px" }}>{k}</div>
                      <div style={{ flex: 1, height: "4px", background: "#0f172a", borderRadius: "2px" }}>
                        <div style={{ width: `${(s[k] || 0) * 100}%`, height: "100%", background: scoreColor(s[k]), borderRadius: "2px" }} />
                      </div>
                      <div style={{ color: "#94a3b8", width: "28px", textAlign: "right" }}>{s[k]?.toFixed(2) || "—"}</div>
                    </div>
                  ))}
                  <div style={{ color: "#64748b", fontSize: "9px", marginTop: "8px", fontStyle: "italic" }}>{s.verdict}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Darwin */}
      {darwinGen > 0 && (
        <div style={{ background: "#0f172a", border: "1px solid #4ade8033", borderRadius: "8px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "10px", color: "#4ade80", letterSpacing: "2px", marginBottom: "8px" }}>🧬 DARWIN EVOLUTION — Generation {darwinGen}/3</div>
          <div style={{ display: "flex", gap: "8px" }}>
            {[1, 2, 3].map(g => (
              <div key={g} style={{ flex: 1, height: "6px", borderRadius: "3px", background: g <= darwinGen ? "#4ade80" : "#1e293b" }} />
            ))}
          </div>
          {winner && <div style={{ marginTop: "10px", fontSize: "11px", color: "#94a3b8" }}>Final score after evolution: <span style={{ color: scoreColor(winner.score?.total_score), fontWeight: 700 }}>{winner.score?.total_score?.toFixed(2)}</span></div>}
        </div>
      )}

      {/* Failure Simulation */}
      {failureModes && (
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", padding: "16px", marginBottom: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <div style={{ fontSize: "10px", color: "#6366f1", letterSpacing: "2px" }}>FAILURE SIMULATION</div>
            <div style={{ fontSize: "10px", background: riskColor(failureModes.overall_risk) + "22", color: riskColor(failureModes.overall_risk), padding: "2px 10px", borderRadius: "10px", border: `1px solid ${riskColor(failureModes.overall_risk)}44` }}>
              OVERALL: {failureModes.overall_risk?.toUpperCase()}
            </div>
          </div>
          <div style={{ display: "grid", gap: "8px" }}>
            {failureModes.failure_modes?.map((f, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "140px 70px 1fr 1fr", gap: "10px", padding: "10px", background: "#1e293b", borderRadius: "4px", fontSize: "10px", alignItems: "start" }}>
                <div style={{ color: "#e2e8f0", fontWeight: 600 }}>{f.scenario}</div>
                <div style={{ color: riskColor(f.risk), background: riskColor(f.risk) + "22", padding: "2px 6px", borderRadius: "3px", textAlign: "center" }}>{f.risk}</div>
                <div style={{ color: "#94a3b8" }}>{f.impact}</div>
                <div style={{ color: "#4ade80" }}>→ {f.mitigation}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Final Report */}
      {phase === "final_report" && winner && (
        <div style={{ background: "#0f172a", border: "2px solid #6366f1", borderRadius: "8px", padding: "20px", marginBottom: "16px" }}>
          <div style={{ fontSize: "10px", color: "#6366f1", letterSpacing: "3px", marginBottom: "16px", textAlign: "center" }}>FINAL REPORT</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "12px", marginBottom: "16px" }}>
            {[
              { label: "WINNER", val: `Patch ${winner.id} (${winner.name})`, color: winner.color },
              { label: "FINAL SCORE", val: winner.score?.total_score?.toFixed(2) || "—", color: scoreColor(winner.score?.total_score) },
              { label: "TESTS", val: `${winner.score?.tests_passed ?? "✓"}/${winner.score?.tests_total ?? "✓"}`, color: "#4ade80" },
              { label: "RISK", val: failureModes?.overall_risk?.toUpperCase() || "—", color: riskColor(failureModes?.overall_risk) },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ textAlign: "center", background: "#1e293b", borderRadius: "6px", padding: "14px" }}>
                <div style={{ fontSize: "9px", color: "#475569", marginBottom: "6px", letterSpacing: "2px" }}>{label}</div>
                <div style={{ fontSize: "16px", fontWeight: 700, color }}>{val}</div>
              </div>
            ))}
          </div>
          <div style={{ background: "#1e293b", borderRadius: "6px", padding: "14px" }}>
            <div style={{ fontSize: "9px", color: "#475569", marginBottom: "8px", letterSpacing: "2px" }}>WINNING PATCH</div>
            <pre style={{ margin: 0, fontSize: "10px", color: "#e2e8f0", whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: "200px", overflowY: "auto" }}>{winner.patch}</pre>
          </div>
          {winner.changes_summary && (
            <div style={{ marginTop: "10px", fontSize: "11px", color: "#94a3b8" }}>
              <span style={{ color: "#475569" }}>Changes: </span>{winner.changes_summary}
            </div>
          )}
          <div style={{ marginTop: "12px", textAlign: "center", fontSize: "14px", color: "#4ade80", fontWeight: 700 }}>✓ PIPELINE COMPLETE — STATUS: SUCCESS</div>
        </div>
      )}

      {/* Memory */}
      {memory && (
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "6px", padding: "12px", fontSize: "10px" }}>
          <div style={{ color: "#475569", marginBottom: "6px", letterSpacing: "2px" }}>📦 .aicos_memory.json</div>
          <pre style={{ margin: 0, color: "#64748b", fontSize: "9px" }}>{JSON.stringify(memory, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
