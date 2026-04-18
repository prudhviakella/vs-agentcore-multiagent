"""
agents/chart/chart_builder.py
==============================
Chart type detection + Chart.js JSON builder.

TODO: implement
  - detect_chart_type(text) -> str   # bar | line | pie | doughnut | scatter
  - extract_data(text) -> dict       # extract labels + datasets from answer
  - build_chartjs_config(data) -> dict
  - build_html_embed(config) -> str  # returns <canvas> + <script> block
"""
