"""
Tool-GRPO: Interactive Evaluation Dashboard Generator
Converts benchmark summary JSON into a production-grade, standalone HTML dashboard.
"""

import json
import os
from typing import Dict, Any, List

def generate_html_eval_dashboard(
    benchmark_reports: List[Dict[str, Any]],
    output_html_path: str,
    title: str = "Tool-GRPO: Reasoning & Tool Alignment Evaluation Dashboard"
) -> str:
    """
    Renders a rich interactive dashboard with metric summaries, error taxonomy,
    category breakdowns, and individual trajectory inspectors.
    """
    reports_json = json.dumps(benchmark_reports, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
    body {{
      font-family: 'Inter', sans-serif;
      background: #090d16;
      color: #e2e8f0;
    }}
    code, pre, .font-mono {{
      font-family: 'JetBrains Mono', monospace;
    }}
    .glow-cyan {{
      box-shadow: 0 0 25px -5px rgba(6, 182, 212, 0.25);
    }}
    .custom-scroll::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    .custom-scroll::-webkit-scrollbar-thumb {{
      background: #334155;
      border-radius: 4px;
    }}
  </style>
</head>
<body class="min-h-screen p-6 custom-scroll">
  <div class="max-w-7xl mx-auto space-y-8">

    <!-- Top Header -->
    <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 gap-4">
      <div>
        <div class="flex items-center gap-3">
          <span class="px-2.5 py-1 bg-cyan-950 border border-cyan-800/80 text-cyan-400 text-xs font-mono rounded-full font-semibold">
            RLVR EVAL SUITE v2.0
          </span>
          <span class="px-2 py-0.5 bg-emerald-950/80 border border-emerald-800/60 text-emerald-400 text-xs font-mono rounded">
            PASS@K & MAJ@K READY
          </span>
        </div>
        <h1 class="text-3xl font-bold tracking-tight text-white mt-2 flex items-center gap-3">
          <i class="fa-solid fa-chart-column text-cyan-400"></i>
          {title}
        </h1>
        <p class="text-sm text-slate-400 mt-1">Multi-Turn Interleaved Tool-GRPO: Empirical Verification & Diagnostic Taxonomy</p>
      </div>
      <div class="flex items-center gap-3">
        <select id="benchmarkSelector" onchange="switchBenchmark(this.value)" class="bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-2 font-mono focus:ring-2 focus:ring-cyan-500 outline-none">
        </select>
        <button onclick="downloadJson()" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-xs font-mono text-slate-200 flex items-center gap-2 transition">
          <i class="fa-solid fa-download"></i> Export JSON
        </button>
      </div>
    </header>

    <!-- Key Metrics Stat Cards -->
    <section class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4" id="statsGrid">
      <!-- Populated dynamically -->
    </section>

    <!-- Secondary Breakdown Grid: Self-Consistency & Failure Taxonomy -->
    <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Pass@k vs Maj@k Card -->
      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 glow-cyan">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
          <i class="fa-solid fa-bullseye text-cyan-400"></i> Reasoning Scaling (Pass@k vs Majority Voting)
        </h2>
        <div id="scalingTable" class="overflow-x-auto">
          <!-- Populated dynamically -->
        </div>
      </div>

      <!-- Failure Taxonomy Breakdown -->
      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 glow-cyan">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
          <i class="fa-solid fa-triangle-exclamation text-amber-400"></i> Failure Mode Taxonomy
        </h2>
        <div id="taxonomyContainer" class="space-y-3">
          <!-- Populated dynamically -->
        </div>
      </div>
    </section>

    <!-- Category Breakdown (e.g. GSM8K vs MATH-500 subjects) -->
    <section class="bg-slate-900/90 border border-slate-800 rounded-xl p-5">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
        <i class="fa-solid fa-layer-group text-purple-400"></i> Subject / Category Performance
      </h2>
      <div id="categoryContainer" class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
        <!-- Populated dynamically -->
      </div>
    </section>

    <!-- Problem-by-Problem Trajectory Inspector -->
    <section class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-list-check text-cyan-400"></i> Problem Evaluation Details
          </h2>
          <p class="text-xs text-slate-400">Inspect interleaved rollouts, sandboxed Python code, execution returns, and verifier comparisons.</p>
        </div>
        <div class="flex items-center gap-2">
          <input type="text" id="filterInput" onkeyup="filterProblems()" placeholder="Filter by keyword / ID..." class="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded px-3 py-1.5 font-mono focus:ring-1 focus:ring-cyan-500 outline-none w-48">
          <select id="statusFilter" onchange="filterProblems()" class="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded px-2 py-1.5 font-mono outline-none">
            <option value="all">All Status</option>
            <option value="correct">Correct Only</option>
            <option value="failed">Failed Only</option>
            <option value="with_tools">Used Tools</option>
          </select>
        </div>
      </div>

      <!-- Problem Accordion List -->
      <div id="problemsList" class="space-y-3 custom-scroll max-h-[700px] overflow-y-auto pr-2">
        <!-- Populated dynamically -->
      </div>
    </section>

  </div>

  <script>
    const benchmarkData = {reports_json};
    let currentBenchmark = benchmarkData[0]?.benchmark_name || "";

    function init() {{
      const selector = document.getElementById("benchmarkSelector");
      selector.innerHTML = "";
      benchmarkData.forEach(b => {{
        const opt = document.createElement("option");
        opt.value = b.benchmark_name;
        opt.textContent = `${{b.benchmark_name}} (${{b.total_problems}} items)`;
        selector.appendChild(opt);
      }});
      if (benchmarkData.length > 0) {{
        renderBenchmark(benchmarkData[0]);
      }}
    }}

    function switchBenchmark(name) {{
      currentBenchmark = name;
      const b = benchmarkData.find(x => x.benchmark_name === name);
      if (b) renderBenchmark(b);
    }}

    function renderBenchmark(b) {{
      // 1. Stat cards
      const grid = document.getElementById("statsGrid");
      grid.innerHTML = `
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <p class="text-xs text-slate-400 uppercase tracking-wide">Pass@1 (Greedy)</p>
          <p class="text-2xl font-bold text-cyan-400 mt-1">${{b.pass_at_1}}%</p>
          <span class="text-[10px] text-slate-500">Deterministic eval</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <p class="text-xs text-slate-400 uppercase tracking-wide">Maj@4 (Cons@4)</p>
          <p class="text-2xl font-bold text-emerald-400 mt-1">${{b.majority_at_k?.['4'] || b.pass_at_1}}%</p>
          <span class="text-[10px] text-slate-500">Majority vote (k=4)</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <p class="text-xs text-slate-400 uppercase tracking-wide">Tool Invocation</p>
          <p class="text-2xl font-bold text-purple-400 mt-1">${{b.tool_invocation_rate}}%</p>
          <span class="text-[10px] text-slate-500">Problems using code</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <p class="text-xs text-slate-400 uppercase tracking-wide">Tool Success</p>
          <p class="text-2xl font-bold text-blue-400 mt-1">${{b.tool_success_rate}}%</p>
          <span class="text-[10px] text-slate-500">Execution code 0</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <p class="text-xs text-slate-400 uppercase tracking-wide">Reflexion Recovery</p>
          <p class="text-2xl font-bold text-amber-400 mt-1">${{b.reflexion_recovery_rate}}%</p>
          <span class="text-[10px] text-slate-500">Self-correction rate</span>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl">
          <p class="text-xs text-slate-400 uppercase tracking-wide">Avg Tool Calls</p>
          <p class="text-2xl font-bold text-rose-400 mt-1">${{b.avg_tool_calls_per_problem}}</p>
          <span class="text-[10px] text-slate-500">Calls / problem</span>
        </div>
      `;

      // 2. Scaling Table
      const kTable = document.getElementById("scalingTable");
      const kKeys = Object.keys(b.pass_at_k || {{}}).sort((x, y) => Number(x) - Number(y));
      if (kKeys.length > 0) {{
        let rows = kKeys.map(k => `
          <tr class="border-b border-slate-800/80 hover:bg-slate-800/40">
            <td class="py-2.5 px-3 font-mono font-bold text-cyan-400">k = ${{k}}</td>
            <td class="py-2.5 px-3 font-mono text-emerald-400 font-semibold">${{b.pass_at_k[k]}}%</td>
            <td class="py-2.5 px-3 font-mono text-purple-400 font-semibold">${{b.majority_at_k?.[k] || '-'}}%</td>
          </tr>
        `).join("");
        kTable.innerHTML = `
          <table class="w-full text-left text-xs">
            <thead>
              <tr class="text-slate-400 border-b border-slate-800">
                <th class="py-2 px-3">Sample Count (k)</th>
                <th class="py-2 px-3">Pass@k (Unbiased)</th>
                <th class="py-2 px-3">Maj@k (Self-Consistency)</th>
              </tr>
            </thead>
            <tbody>${{rows}}</tbody>
          </table>
        `;
      }} else {{
        kTable.innerHTML = `<p class="text-xs text-slate-500 py-4">Single sample mode evaluated (k=1).</p>`;
      }}

      // 3. Taxonomy
      const tax = document.getElementById("taxonomyContainer");
      const failKeys = Object.keys(b.failure_taxonomy || {{}});
      if (failKeys.length > 0) {{
        const totalFails = failKeys.reduce((acc, k) => acc + b.failure_taxonomy[k], 0);
        tax.innerHTML = failKeys.map(k => {{
          const count = b.failure_taxonomy[k];
          const pct = Math.round((count / totalFails) * 100);
          return `
            <div>
              <div class="flex justify-between text-xs mb-1">
                <span class="font-mono text-slate-300 font-medium">${{k}}</span>
                <span class="font-mono text-slate-400">${{count}} (${{pct}}%)</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-2">
                <div class="bg-amber-500 h-2 rounded-full" style="width: ${{pct}}%"></div>
              </div>
            </div>
          `;
        }}).join("");
      }} else {{
        tax.innerHTML = `<p class="text-xs text-emerald-400 py-4"><i class="fa-solid fa-circle-check"></i> Zero failure modes recorded.</p>`;
      }}

      // 4. Categories
      const catCont = document.getElementById("categoryContainer");
      const catKeys = Object.keys(b.category_metrics || {{}});
      if (catKeys.length > 0) {{
        catCont.innerHTML = catKeys.map(k => {{
          const cat = b.category_metrics[k];
          return `
            <div class="bg-slate-950 border border-slate-800 p-3 rounded-lg">
              <p class="text-xs font-mono font-semibold text-slate-300 uppercase">${{k}}</p>
              <div class="flex justify-between items-end mt-2">
                <div>
                  <span class="text-xs text-slate-500 block">Pass@1</span>
                  <span class="text-lg font-bold text-cyan-400 font-mono">${{cat.pass_at_1}}%</span>
                </div>
                <div>
                  <span class="text-xs text-slate-500 block">Maj@k</span>
                  <span class="text-lg font-bold text-emerald-400 font-mono">${{cat.majority_correct}}%</span>
                </div>
                <div class="text-right">
                  <span class="text-[10px] text-slate-500 block">Samples</span>
                  <span class="text-xs font-mono text-slate-400">${{cat.count}}</span>
                </div>
              </div>
            </div>
          `;
        }}).join("");
      }} else {{
        catCont.innerHTML = `<p class="text-xs text-slate-500">No category breakdown available.</p>`;
      }}

      // 5. Problems List
      renderProblemsList(b.sample_results || []);
    }}

    function renderProblemsList(items) {{
      const list = document.getElementById("problemsList");
      list.innerHTML = items.map((item, idx) => {{
        const isPass1 = item.correct_mask?.[0] || false;
        const isMaj = item.majority_correct || false;
        const statusBadge = isPass1 
          ? `<span class="px-2 py-0.5 bg-emerald-950 text-emerald-400 border border-emerald-800 rounded text-[11px] font-mono">PASS@1</span>`
          : `<span class="px-2 py-0.5 bg-rose-950 text-rose-400 border border-rose-800 rounded text-[11px] font-mono">FAIL</span>`;
        
        const majBadge = isMaj 
          ? `<span class="px-2 py-0.5 bg-cyan-950 text-cyan-400 border border-cyan-800 rounded text-[11px] font-mono">MAJ: WIN (${{item.majority_votes}}v)</span>`
          : `<span class="px-2 py-0.5 bg-slate-800 text-slate-400 rounded text-[11px] font-mono">MAJ: FAIL</span>`;

        const toolBadge = item.tool_calls > 0
          ? `<span class="px-2 py-0.5 bg-purple-950 text-purple-300 border border-purple-800 rounded text-[11px] font-mono"><i class="fa-solid fa-terminal text-[9px]"></i> ${{item.tool_calls}} TOOLS</span>`
          : `<span class="px-2 py-0.5 bg-slate-800 text-slate-500 rounded text-[11px] font-mono">DIRECT</span>`;

        const failBadge = item.failure_mode
          ? `<span class="px-2 py-0.5 bg-amber-950 text-amber-300 border border-amber-800 rounded text-[11px] font-mono">${{item.failure_mode}}</span>`
          : '';

        const completionText = item.trajectories?.[0]?.completion || item.trajectories?.[0]?.text || "No completion";

        return `
          <div class="problem-card bg-slate-950 border border-slate-800/90 rounded-lg p-3 hover:border-slate-700 transition" 
               data-id="${{item.problem_id}}" data-correct="${{isPass1}}" data-tools="${{item.tool_calls > 0}}" data-text="${{(item.question + ' ' + item.problem_id).toLowerCase()}}">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 cursor-pointer" onclick="toggleDetails('detail-${{idx}}')">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="font-mono text-xs font-bold text-slate-400">#${{item.problem_id}}</span>
                ${{statusBadge}}
                ${{majBadge}}
                ${{toolBadge}}
                ${{failBadge}}
                <span class="text-xs text-slate-500 font-mono">[${{item.category}}]</span>
              </div>
              <div class="flex items-center gap-3 text-xs font-mono">
                <span class="text-slate-400">Pred: <strong class="text-white">${{item.majority_prediction || item.predictions?.[0] || 'None'}}</strong></span>
                <span class="text-slate-400">GT: <strong class="text-emerald-400">${{item.ground_truth}}</strong></span>
                <i class="fa-solid fa-chevron-down text-slate-500 text-xs transition-transform" id="chevron-detail-${{idx}}"></i>
              </div>
            </div>

            <p class="text-xs text-slate-300 mt-2 line-clamp-2">${{item.question}}</p>

            <!-- Collapsible Trajectory Details -->
            <div id="detail-${{idx}}" class="hidden mt-4 pt-3 border-t border-slate-800/80 space-y-3">
              <div>
                <p class="text-[11px] font-mono uppercase text-slate-400 mb-1">Full Problem Prompt:</p>
                <div class="bg-slate-900 p-2.5 rounded text-xs font-mono text-slate-300 whitespace-pre-wrap">${{item.question}}</div>
              </div>

              <div>
                <p class="text-[11px] font-mono uppercase text-slate-400 mb-1">Trajectory Sample #1 (Interleaved Execution):</p>
                <div class="bg-slate-900 border border-slate-800 p-3 rounded text-xs font-mono text-slate-300 whitespace-pre-wrap max-h-72 overflow-y-auto custom-scroll">${{escapeHtml(completionText)}}</div>
              </div>

              <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono bg-slate-900/60 p-2 rounded">
                <div><span class="text-slate-500">Tool Calls:</span> <span class="text-slate-200">${{item.tool_calls}}</span></div>
                <div><span class="text-slate-500">Exec Errors:</span> <span class="text-slate-200">${{item.exec_failures}}</span></div>
                <div><span class="text-slate-500">Repairs:</span> <span class="text-slate-200">${{item.repairs}}</span></div>
                <div><span class="text-slate-500">Avg Tokens:</span> <span class="text-slate-200">${{item.avg_tokens}}</span></div>
              </div>
            </div>
          </div>
        `;
      }}).join("");
    }}

    function toggleDetails(id) {{
      const el = document.getElementById(id);
      const chev = document.getElementById("chevron-" + id);
      if (el) {{
        el.classList.toggle("hidden");
        if (chev) chev.classList.toggle("rotate-180");
      }}
    }}

    function filterProblems() {{
      const query = document.getElementById("filterInput").value.toLowerCase();
      const status = document.getElementById("statusFilter").value;
      const cards = document.querySelectorAll(".problem-card");

      cards.forEach(card => {{
        const text = card.getAttribute("data-text");
        const correct = card.getAttribute("data-correct") === "true";
        const tools = card.getAttribute("data-tools") === "true";

        let matchesText = !query || text.includes(query);
        let matchesStatus = true;
        if (status === "correct") matchesStatus = correct;
        if (status === "failed") matchesStatus = !correct;
        if (status === "with_tools") matchesStatus = tools;

        if (matchesText && matchesStatus) {{
          card.classList.remove("hidden");
        }} else {{
          card.classList.add("hidden");
        }}
      }});
    }}

    function escapeHtml(text) {{
      return (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    function downloadJson() {{
      const blob = new Blob([JSON.stringify(benchmarkData, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "tool_grpo_eval_report.json";
      a.click();
      URL.revokeObjectURL(url);
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_html_path) or ".", exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path
