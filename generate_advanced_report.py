"""
generate_advanced_report.py
Run: python3 generate_advanced_report.py
Produces: _report_data.json + report_data.js (feeds advanced_report.html)
"""
import json, pathlib
# Reuse the analysis module
import _advanced_analysis as aa  # executes full pipeline, writes _report_data.json

data = json.load(open("_report_data.json"))
pathlib.Path("report_data.js").write_text("var REPORT_DATA = " + json.dumps(data, indent=2) + ";\n")
print("✓ Report regenerated.")
print(f"  Trades: {len(data['trade_diag'])} | Sweep: {len(data['sweep_top12'])} | Recommendations: {len(data['recommendations'])}")
