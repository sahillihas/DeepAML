# DeepAML - Deep Anti-Money Laundering Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

DeepAML is a state-of-the-art anti-money laundering (AML) detection system that combines advanced graph analytics with deep learning to identify sophisticated money laundering patterns. By analyzing complex transaction networks and behavioral patterns, the system provides financial institutions with powerful tools to combat financial crime.

Key capabilities:
- Real-time transaction monitoring
- Advanced pattern recognition using graph neural networks
- Automated risk assessment and case generation
- Regulatory compliance support (FinCEN, FATF guidelines)

## Features

- **Data Ingestion & Processing**
  - Multiple data source support (JSON, CSV, DataFrames, SQL databases)
  - Real-time streaming capabilities via Apache Kafka
  - Automated data cleaning and standardization
  - Entity resolution and deduplication

- **Knowledge Graph Construction**
  - Flexible graph backend (NetworkX/Neo4j)
  - Temporal transaction modeling
  - Entity relationship mapping
  - Hierarchical account structure support

- **Advanced Analytics**
  - Graph Neural Network (GNN) based pattern detection
  - Temporal pattern analysis
  - Behavioral profiling
  - Anomaly detection using embedding spaces
  
- **Pattern Detection**
  - Cycle Detection (Round-tripping)
  - Structuring Patterns (Smurfing)
  - Layering Schemes
  - Fan-in/Fan-out Analysis
  - Rapid Movement Detection
  - Shell Company Pattern Recognition

- **Risk Scoring**: Entity-level and transaction-level risk assessment
- **Case Management**: Automated case generation and management
- **Reporting**: Detailed reports and Suspicious Activity Report (SAR) generation

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepAML.git
cd DeepAML
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your transaction data in JSON format:
```json
[
    {
        "transaction_id": "TX001",
        "sender_account": "ACC001",
        "sender_name": "John Doe",
        "sender_bank": "Bank A",
        "sender_country": "US",
        "recipient_account": "ACC002",
        "recipient_name": "Jane Smith",
        "recipient_bank": "Bank B",
        "recipient_country": "UK",
        "amount": 50000,
        "currency": "USD",
        "timestamp": "2023-01-01T10:00:00"
    }
]
```

2. Run the system:
```bash
python main.py
```

## Configuration

- Neo4j Configuration (Optional):
  - Set environment variables for Neo4j connection:
    ```bash
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="your_password"
    ```

## Output

The system generates:
1. Summary reports of detected patterns
2. Risk scores for entities
3. Suspicious Activity Reports (SARs) for high-risk cases
4. Visualization of transaction networks (when using NetworkX)

## Architecture

```
DeepAML/
├── core/              # Core system components
├── models/            # ML/DL models
├── graph_ops/         # Graph operations
├── data_handlers/     # Data processing
├── api/              # REST API
└── utils/            # Utility functions
```

## Implementation Details

The system implements several key algorithms:
- Graph Neural Networks (GNN) for pattern recognition
- Temporal Graph Attention Networks
- Node2Vec for entity embedding
- LSTM-based anomaly detection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Thanks
## Citation

If you use DeepAML in your research, please cite:
```bibtex
@software{deepaml2023,
  author = {Your Name},
  title = {DeepAML: Deep Learning-based Anti-Money Laundering Detection},
  year = {2023},
  url = {https://github.com/yourusername/DeepAML}
}
```
