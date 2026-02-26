"""
Feature engineering package for insider trading analysis.

Modules
-------
trade_features    – Per-trade transforms (log value, direction, ownership %)
insider_features  – Insider-level aggregate stats and role encoding
temporal_features – Rolling-window and time-series features
network_features  – Cross-insider coordination detection
text_features     – Footnote NLP (keyword flags, routine-language score)
pipeline          – Orchestrates all modules and writes the final feature CSV
"""
