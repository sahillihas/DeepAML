# DeepAML: Real-time Anti-Money Laundering Detection System

## Overview
DeepAML is a real-time anti-money laundering detection system that combines pattern recognition with live monitoring capabilities. The system can both detect suspicious patterns in real-time and simulate various types of money laundering attacks for testing and validation purposes.

## Key Features

### Real-time Monitoring
- Live transaction monitoring with instant alerts for high-confidence patterns (>90% confidence)
- Periodic analysis (configurable interval, default 60 seconds) for complex pattern detection
- Support for both immediate and batch pattern detection
- Continuous file monitoring with efficient transaction processing

### Pattern Detection
- **Structuring (Smurfing)**: Detection of multiple small transactions to avoid reporting thresholds
- **Layering**: Identification of complex transaction chains through multiple intermediaries
- **Round-Trip**: Detection of funds returning to origin through multiple hops
- **Rapid Movement**: Identification of quick successive transfers
- **Fan-in/Fan-out**: Detection of many-to-one and one-to-many transaction patterns

### Attack Simulation
- Interactive command-line interface for attack pattern generation
- Support for both immediate and scheduled attacks
- Detailed attack reporting with transaction flows and risk metrics
- Multiple attack patterns supported:
  - Structuring: Break large amounts into smaller transactions
  - Layering: Create complex chains of transactions
  - Round-Trip: Generate circular transaction patterns
  - Rapid Movement: Create quick successive transfers
  - Fan-in: Simulate multiple sources to one destination
  - Fan-out: Simulate one source to multiple destinations

### Risk Analysis
- Real-time risk scoring for individual transactions
- Pattern-specific risk assessment
- Composite risk calculation based on multiple factors
- Dynamic thresholding based on pattern types

## Usage

### Real-time Monitoring
```bash
python src/main.py --mode monitor --file /path/to/transactions.json [options]

Options:
  --risk-threshold FLOAT    Risk threshold for alerting (0.0-1.0, default: 0.7)
  --check-interval INT      Interval in seconds for periodic checks (default: 60)
  --log-level LEVEL        Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
```

### Attack Simulation
```bash
python src/main.py --mode simulate --file /path/to/transactions.json

Interactive menu options:
1. Structuring Pattern
2. Layering Pattern
3. Round-Trip Pattern
4. Rapid Movement Pattern
5. Fan-In Pattern
6. Fan-Out Pattern
7. Random Pattern
8. Schedule an Attack
9. View Scheduled Attacks
10. Exit
```

## Transaction Format
The system expects transactions in JSON format:
```json
{
  "transactions": [
    {
      "transaction_id": "TX123",
      "timestamp": "2025-05-03T10:30:00",
      "amount": 5000.00,
      "currency": "USD",
      "sender": "ACC001",
      "sender_name": "Entity_001",
      "sender_bank": "Bank_A",
      "sender_country": "US",
      "receiver": "ACC002",
      "receiver_name": "Entity_002",
      "receiver_bank": "Bank_B",
      "receiver_country": "GB",
      "reference": "Payment_123"
    }
  ]
}
```

## Pattern Detection Methods

### 1. Structuring Detection
- Monitors transactions below reporting thresholds
- Groups transactions by sender-receiver pairs
- Analyzes transaction patterns within configurable time windows
- Risk factors: total amount, number of transactions, time span

### 2. Layering Detection
- Builds transaction graph for path analysis
- Identifies complex transaction chains
- Considers intermediary accounts and jurisdictions
- Risk factors: number of hops, jurisdictions involved, amount variations

### 3. Round-Trip Detection
- Identifies circular transaction patterns
- Analyzes fund flow returning to origin
- Considers time windows and amount variations
- Risk factors: cycle length, time to complete, amount differences

### 4. Rapid Movement Detection
- Monitors transaction velocity
- Identifies quick successive transfers
- Analyzes transaction chains and timing
- Risk factors: transfer speed, number of hops, amount patterns

### 5. Fan Patterns Detection
- Analyzes many-to-one (fan-in) and one-to-many (fan-out) patterns
- Considers transaction timing and amounts
- Monitors source/destination diversity
- Risk factors: number of participants, amount distribution, geographic spread

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepAML.git
cd DeepAML
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
