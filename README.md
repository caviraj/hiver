# Hiver - Customer Support AI Agent

An enterprise AI agent for automated customer support, conversation graph reconstruction, intent classification, and resolution routing.

## Overview

This repository houses the end-to-end intelligence and data processing pipeline for customer support interaction analysis.

### Milestone 1: Data Infrastructure & Preprocessing
- **Phase 1.1: Ingestion & Graph Reconstruction**
  - `M1.P1.1.F1`: Dataset Ingestion & Brand Filtering
  - `M1.P1.1.F2`: Conversation Thread Graph Reconstruction (Upcoming)
  - `M1.P1.1.F3`: Text Normalization & PII Masking (Upcoming)

## Project Structure

```text
hiver/
├── config/
│   └── data_config.yaml          # Dataset ingestion & brand filtering configuration
├── src/
│   └── data/
│       ├── schema.py             # Pydantic schemas (RawTweet, IngestStats)
│       ├── validators.py         # Row-level validation & rejection logging
│       └── ingest.py             # Dataset loading, validation, and brand filtering
├── tests/
│   └── data/
│       └── test_ingest.py        # Comprehensive test suite
├── .gitignore                    # Git ignore rules
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- `pip`

### Installation
```bash
pip install pydantic pyyaml pytest
```

### Running Tests
```bash
pytest
```
