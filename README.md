# DeepAML: A Real-Time Anti-Money Laundering Detection System

## 🚀 Overview

**DeepAML** is a real-time system designed to detect and analyse suspicious financial activities, particularly those related to money laundering. It combines live transaction monitoring with smart pattern recognition and even includes a simulation engine to help you test how the system handles different types of laundering strategies.

Whether you're working on real-world AML applications or building a proof-of-concept, DeepAML is designed to give you the tools to both detect and simulate fraudulent behaviours efficiently.

---

## 🔍 Key Features

### 🕵️ Real-Time Monitoring

* Instantly detects and alerts on suspicious patterns with high confidence (90%+).
* Periodic checks (default every 60 seconds) to uncover more complex behaviour.
* Supports both real-time and batch analysis of transactions.
* Monitors incoming files continuously for seamless transaction tracking.

### 🧠 Auto Pattern Detection

DeepAML is built to spot common laundering techniques like:

* **Structuring (Smurfing)**: Breaking large transactions into smaller ones to avoid detection.
* **Layering**: Moving money through multiple accounts to obscure its origin.
* **Round-Trip**: Sending funds out and bringing them back through a chain of hops.
* **Rapid Movement**: Quick successive transactions indicating potential laundering.
* **Fan-In / Fan-Out**: Many-to-one or one-to-many transaction patterns, often used to obscure trails.

### 🧪 Attack Simulation

Test the system’s detection capabilities with built-in laundering scenario simulations:

* Interactive CLI interface for generating patterns.
* Choose between immediate or scheduled simulations.
* Detailed reports including risk scores and transaction flow diagrams.

Available simulations include:

* Structuring
* Layering
* Round-Trip
* Rapid Movement
* Fan-In and Fan-Out
* Random combinations
* Scheduling future simulations

### 📊 Risk Scoring Engine

* Assigns risk scores to individual transactions in real-time.
* Calculates pattern-specific and combined risk metrics.
* Supports adaptive thresholds depending on the pattern type detected.

---

## 🛠️ How to Use

### Real-Time Monitoring

Monitor a file with transactions and receive alerts in real time:

```bash
python src/main.py --mode monitor --file /path/to/transactions.json
```

Optional parameters:

* `--risk-threshold FLOAT` – Set the minimum risk score to trigger an alert (default: 0.7)
* `--check-interval INT` – Frequency of periodic scans in seconds (default: 60)
* `--log-level LEVEL` – Logging verbosity (DEBUG, INFO, WARNING, etc.)

### Running Simulations

Launch the CLI tool to simulate laundering scenarios:

```bash
python src/main.py --mode simulate --file /path/to/transactions.json
```

From there, you’ll see an interactive menu like:

```
1. Structuring
2. Layering
3. Round-Trip
4. Rapid Movement
5. Fan-In
6. Fan-Out
7. Random Pattern
8. Schedule an Attack
9. View Scheduled Attacks
10. Exit
```

---

## 📄 Transaction Format

DeepAML works with transactions formatted as JSON. Here’s an example:

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

---

## 🧩 How Detection Works

### Structuring

* Identifies multiple small transactions sent to avoid thresholds.
* Groups data by sender/receiver and analyzes over a time window.
* Risk factors: total sum, frequency, and spread over time.

### Layering

* Creates a transaction graph to follow the money trail.
* Flags long or complex chains using multiple intermediaries.
* Risk factors: number of hops, jurisdictions involved, amount variation.

### Round-Trip

* Detects funds sent out and returned via different routes.
* Looks for cyclical transaction paths.
* Risk factors: loop length, timing, and amounts.

### Rapid Movement

* Flags accounts with high transaction velocity.
* Evaluates both chain length and timing.
* Risk factors: speed, hop count, and transaction consistency.

### Fan-In / Fan-Out

* Fan-In: Many sources sending to a single account.
* Fan-Out: One source dispersing funds to many recipients.
* Risk factors: volume of accounts involved, amount spread, geography.

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/DeepAML.git
cd DeepAML
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on our code of conduct and how to submit pull requests.

---

## 📜 License

DeepAML is released under the MIT License. See the `LICENSE` file for more information.

---
