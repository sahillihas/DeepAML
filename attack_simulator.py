"""
Attack Simulator for AML Testing
------------------------------
Simulates various money laundering patterns by injecting transactions into the monitoring system
"""

import json
import random
import time
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import sys
from queue import PriorityQueue
from colorama import init, Fore, Style
import numpy as np
from models import PatternType, RiskLevel

# Initialize colorama
init()

# Constants
HIGH_RISK_COUNTRIES = ['CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG']
MEDIUM_RISK_COUNTRIES = ['CH', 'HK', 'SG', 'MO', 'BS', 'AI']
LOW_RISK_COUNTRIES = ['US', 'GB', 'DE', 'FR', 'CA', 'AU', 'JP', 'NL']
STRUCTURING_THRESHOLD = 10000

class AttackSimulator:
    def __init__(self):
        """Initialize attack simulator"""
        self.accounts = {}
        self.transaction_file = "data/raw/transactions.json"
        self.attack_details_dir = "data/flagged_attacks"
        
        # Initialize scheduler components
        self.scheduled_attacks = PriorityQueue()
        self.running = False
        self.scheduler_thread = None
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.transaction_file), exist_ok=True)
        os.makedirs(self.attack_details_dir, exist_ok=True)
        
        # Initialize empty transaction file if it doesn't exist
        if not os.path.exists(self.transaction_file):
            with open(self.transaction_file, 'w') as f:
                json.dump({'transactions': []}, f)
                
        # Load existing transactions
        self.load_transactions()
        
        # Start scheduler automatically
        self.start_scheduler()

    def load_transactions(self):
        """Load transactions from file"""
        try:
            with open(self.transaction_file, 'r') as f:
                data = json.load(f)
                self.transactions = data.get('transactions', [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.transactions = []

    def save_transactions(self):
        """Save transactions to file"""
        with open(self.transaction_file, 'w') as f:
            json.dump({'transactions': self.transactions}, f, indent=2)

    def generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        return f"TX{uuid.uuid4().hex[:12].upper()}"

    def _get_or_create_account(self, risk_level: RiskLevel) -> Dict:
        """Get or create an account with specified risk level"""
        cache_key = f"{risk_level.value}_{len(self.accounts)}"
        if cache_key in self.accounts:
            return self.accounts[cache_key]

        account = {
            'account_id': f"ACC{str(len(self.accounts)).zfill(6)}",
            'name': f"Entity_{str(len(self.accounts)).zfill(6)}",
            'bank': f"Bank_{random.randint(1, 50)}",
            'country': random.choice({
                RiskLevel.HIGH: HIGH_RISK_COUNTRIES,
                RiskLevel.MEDIUM: MEDIUM_RISK_COUNTRIES,
                RiskLevel.LOW: LOW_RISK_COUNTRIES
            }[risk_level]),
            'risk_level': risk_level.value
        }
        self.accounts[cache_key] = account
        return account

    def create_transaction(self, sender: Dict, receiver: Dict, amount: float, 
                         timestamp: Optional[datetime] = None, 
                         pattern_type: Optional[PatternType] = None,
                         metadata: Dict = None) -> Dict:
        """Create a transaction with enhanced metadata"""
        if timestamp is None:
            timestamp = datetime.now()

        tx = {
            'transaction_id': self.generate_transaction_id(),
            'timestamp': timestamp.isoformat(),
            'amount': round(amount, 2),
            'currency': random.choice(['USD', 'EUR', 'GBP']),
            'sender': sender['account_id'],
            'sender_name': sender['name'],
            'sender_bank': sender['bank'],
            'sender_country': sender['country'],
            'receiver': receiver['account_id'],
            'receiver_name': receiver['name'],
            'receiver_bank': receiver['bank'],
            'receiver_country': receiver['country'],
            'reference': f"Payment_{uuid.uuid4().hex[:8]}"
        }

        if pattern_type:
            tx['pattern_type'] = pattern_type.value
            
        if metadata:
            tx['metadata'] = metadata

        return tx

    def _print_attack_details(self, attack_type: str, transactions: List[Dict], additional_info: Dict = None):
        """Print summarized attack information and save to files"""
        # Get first and last transaction for source/destination
        first_tx = transactions[0]
        last_tx = transactions[-1]
        total_amount = sum(tx['amount'] for tx in transactions)
        
        # Format main summary in a compact way
        print(f"{Fore.RED}🚨 {attack_type} Detected{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Flow:{Style.RESET_ALL} {first_tx['sender']} → {last_tx['receiver']}")
        print(f"{Fore.YELLOW}Amount:{Style.RESET_ALL} ${total_amount:,.2f} | {Fore.YELLOW}Steps:{Style.RESET_ALL} {len(transactions)}")
        
        if additional_info:
            print(f"{Fore.GREEN}Details:{Style.RESET_ALL}")
            for key, value in additional_info.items():
                print(f"  • {key}: {value}")

        # Save transactions to monitored file
        self.save_transactions()
        print(f"\n💾 Written to {os.path.basename(self.transaction_file)}")
        
        # Save attack details to flagged_attacks directory
        output_dir = "data/flagged_attacks"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{attack_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        attack_data = {
            "attack_type": attack_type,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_amount": total_amount,
                "num_transactions": len(transactions),
                "source": {
                    "account": first_tx['sender'],
                    "name": first_tx['sender_name'],
                    "country": first_tx['sender_country']
                },
                "destination": {
                    "account": last_tx['receiver'],
                    "name": last_tx['receiver_name'],
                    "country": last_tx['receiver_country']  
                },
                "time_span": self._calculate_time_span(transactions)
            },
            "transactions": transactions,
            "additional_info": additional_info
        }
        
        with open(output_file, "w") as f:
            json.dump(attack_data, f, indent=2)
        print(f"📁 Details saved to: {os.path.basename(output_file)}")

    def _calculate_time_span(self, transactions: List[Dict]) -> str:
        """Calculate human readable time span between first and last transaction"""
        start_time = datetime.fromisoformat(transactions[0]['timestamp'])
        end_time = datetime.fromisoformat(transactions[-1]['timestamp'])
        delta = end_time - start_time
        
        if delta.days > 0:
            return f"{delta.days} days, {delta.seconds//3600} hours"
        elif delta.seconds >= 3600:
            return f"{delta.seconds//3600} hours, {(delta.seconds%3600)//60} minutes"
        else:
            return f"{delta.seconds//60} minutes"

    def simulate_fan_in(self, target_amount: float, num_senders: int) -> List[Dict]:
        """Simulate fan-in pattern (multiple senders to one receiver)"""
        transactions = []
        receiver = self._get_or_create_account(RiskLevel.HIGH)
        amounts = self._split_amount(target_amount, num_senders)
        base_time = datetime.now()

        for i, amount in enumerate(amounts):
            sender = self._get_or_create_account(RiskLevel.MEDIUM)
            tx_time = base_time + timedelta(hours=random.uniform(0, 48))
            tx = self.create_transaction(
                sender, receiver, amount, tx_time,
                pattern_type=PatternType.FAN_IN
            )
            transactions.append(tx)

        self._print_attack_details(
            "Fan-In Pattern",
            transactions,
            {
                "Target Account": receiver['account_id'],
                "Number of Sources": num_senders,
                "Average Amount": f"{target_amount/num_senders:,.2f}"
            }
        )
        return transactions

    def simulate_fan_out(self, source_amount: float, num_receivers: int) -> List[Dict]:
        """Simulate fan-out pattern (one sender to multiple receivers)"""
        transactions = []
        sender = self._get_or_create_account(RiskLevel.HIGH)
        amounts = self._split_amount(source_amount, num_receivers)
        base_time = datetime.now()

        for i, amount in enumerate(amounts):
            receiver = self._get_or_create_account(RiskLevel.MEDIUM)
            tx_time = base_time + timedelta(hours=random.uniform(0, 48))
            tx = self.create_transaction(
                sender, receiver, amount, tx_time,
                pattern_type=PatternType.FAN_OUT
            )
            transactions.append(tx)

        self._print_attack_details(
            "Fan-Out Pattern",
            transactions,
            {
                "Source Account": sender['account_id'],
                "Number of Recipients": num_receivers,
                "Average Amount": f"{source_amount/num_receivers:,.2f}"
            }
        )
        return transactions

    def simulate_structuring(self, total_amount: float, num_transactions: int) -> List[Dict]:
        """Simulate structuring pattern with enhanced detection evasion"""
        transactions = []
        sender = self._get_or_create_account(RiskLevel.MEDIUM)
        receiver = self._get_or_create_account(RiskLevel.MEDIUM)
        base_time = datetime.now()
        
        # Calculate amounts to stay just under threshold
        amounts = []
        remaining = total_amount
        while remaining > 0:
            if remaining > STRUCTURING_THRESHOLD:
                amount = random.uniform(STRUCTURING_THRESHOLD * 0.85, STRUCTURING_THRESHOLD * 0.95)
            else:
                amount = remaining
            amounts.append(amount)
            remaining -= amount

        # Create transactions with realistic time gaps
        for i, amount in enumerate(amounts):
            tx_time = base_time + timedelta(
                hours=random.uniform(i*12, (i+1)*24)  # Space out over days
            )
            tx = self.create_transaction(
                sender, receiver, amount, tx_time,
                pattern_type=PatternType.STRUCTURING
            )
            transactions.append(tx)

        self._print_attack_details(
            "Structuring Pattern",
            transactions,
            {
                "Original Amount": f"{total_amount:,.2f}",
                "Average Transaction": f"{total_amount/len(transactions):,.2f}",
                "Threshold Avoided": f"{STRUCTURING_THRESHOLD:,.2f}"
            }
        )
        return transactions

    def simulate_layering(self, amount: float, num_layers: int) -> List[Dict]:
        """Simulate complex layering pattern with multiple paths"""
        transactions = []
        base_time = datetime.now()
        
        # Create multiple paths between source and destination
        num_paths = random.randint(2, 3)
        source = self._get_or_create_account(RiskLevel.HIGH)
        final_destination = self._get_or_create_account(RiskLevel.HIGH)
        
        path_amounts = self._split_amount(amount, num_paths)
        
        for path_idx, path_amount in enumerate(path_amounts):
            prev_account = source
            path_layers = []
            
            # Create intermediary accounts for each layer
            for layer in range(num_layers):
                next_account = self._get_or_create_account(
                    RiskLevel.MEDIUM if layer < num_layers-1 else RiskLevel.HIGH
                )
                path_layers.append(next_account)
            
            # Add final destination
            path_layers.append(final_destination)
            
            # Create transactions between layers with slight amount variations
            for layer_idx, next_account in enumerate(path_layers):
                layer_amount = path_amount * random.uniform(0.95, 1.05)
                tx_time = base_time + timedelta(
                    hours=random.uniform(
                        (path_idx * 24) + (layer_idx * 12),
                        (path_idx * 24) + ((layer_idx + 1) * 12)
                    )
                )
                tx = self.create_transaction(
                    prev_account, next_account, layer_amount, tx_time,
                    pattern_type=PatternType.LAYERING
                )
                transactions.append(tx)
                prev_account = next_account

        self._print_attack_details(
            "Layering Pattern",
            transactions,
            {
                "Initial Amount": f"{amount:,.2f}",
                "Number of Paths": num_paths,
                "Layers per Path": num_layers,
                "Total Transactions": len(transactions)
            }
        )
        return transactions

    def simulate_round_trip(self, amount: float, num_hops: int) -> List[Dict]:
        """Simulate round-trip transactions with enhanced complexity"""
        transactions = []
        base_time = datetime.now()
        
        # Create circular chain of accounts
        accounts = [self._get_or_create_account(RiskLevel.HIGH)]  # Start with high risk
        for _ in range(num_hops - 1):
            accounts.append(self._get_or_create_account(RiskLevel.MEDIUM))
        accounts.append(accounts[0])  # Complete the circle
        
        # Create transactions through the chain with varying amounts
        current_amount = amount
        for i in range(len(accounts) - 1):
            # Vary amount slightly in each hop
            current_amount *= random.uniform(0.98, 1.02)
            tx_time = base_time + timedelta(hours=random.uniform(i*24, (i+1)*24))
            
            tx = self.create_transaction(
                accounts[i], accounts[i+1], current_amount, tx_time,
                pattern_type=PatternType.ROUND_TRIP
            )
            transactions.append(tx)

        self._print_attack_details(
            "Round-Trip Pattern",
            transactions,
            {
                "Initial Amount": f"{amount:,.2f}",
                "Final Amount": f"{current_amount:,.2f}",
                "Number of Hops": num_hops,
                "Time to Complete": f"{(len(transactions)-1)*24} hours"
            }
        )
        return transactions

    def simulate_rapid_movement(self, amount: float, num_transactions: int) -> List[Dict]:
        """Simulate rapid movement pattern with quick transfers"""
        transactions = []
        base_time = datetime.now()
        
        # Create chain of accounts
        accounts = [self._get_or_create_account(RiskLevel.HIGH)]
        for _ in range(num_transactions):
            accounts.append(self._get_or_create_account(RiskLevel.HIGH))
            
        # Create rapid transactions through the chain
        current_amount = amount
        for i in range(len(accounts) - 1):
            # Vary amount slightly
            current_amount *= random.uniform(0.99, 1.01)
            # Transactions happen within minutes of each other
            tx_time = base_time + timedelta(minutes=random.uniform(i*15, (i+1)*30))
            
            tx = self.create_transaction(
                accounts[i], accounts[i+1], current_amount, tx_time,
                pattern_type=PatternType.RAPID_MOVEMENT
            )
            transactions.append(tx)

        self._print_attack_details(
            "Rapid Movement Pattern",
            transactions,
            {
                "Initial Amount": f"{amount:,.2f}",
                "Final Amount": f"{current_amount:,.2f}",
                "Number of Movements": num_transactions,
                "Average Time Between Transfers": "15-30 minutes"
            }
        )
        return transactions

    def _split_amount(self, total: float, num_parts: int) -> List[float]:
        """Split total amount into random parts that sum to total"""
        splits = sorted([random.random() for _ in range(num_parts - 1)])
        splits = [0] + splits + [1]
        amounts = [total * (splits[i+1] - splits[i]) for i in range(len(splits)-1)]
        random.shuffle(amounts)
        return amounts

    def simulate_random_attack(self) -> List[Dict]:
        """Simulate a random attack pattern"""
        pattern = random.choice(list(PatternType))
        amount = random.uniform(50000, 500000)
        
        if pattern == PatternType.STRUCTURING:
            num_tx = random.randint(5, 10)
            return self.simulate_structuring(amount, num_tx)
            
        elif pattern == PatternType.LAYERING:
            num_layers = random.randint(3, 6)
            return self.simulate_layering(amount, num_layers)
            
        elif pattern == PatternType.ROUND_TRIP:
            num_hops = random.randint(3, 7)
            return self.simulate_round_trip(amount, num_hops)
            
        elif pattern == PatternType.RAPID_MOVEMENT:
            num_tx = random.randint(5, 8)
            return self.simulate_rapid_movement(amount, num_tx)
            
        elif pattern == PatternType.FAN_IN:
            num_senders = random.randint(5, 12)
            return self.simulate_fan_in(amount, num_senders)
            
        else:  # FAN_OUT
            num_receivers = random.randint(5, 12)
            return self.simulate_fan_out(amount, num_receivers)

    def generate_attacks(self, pattern_types: List[PatternType] = None) -> List[Dict]:
        """Generate suspicious transactions based on specified patterns"""
        if pattern_types is None:
            pattern_types = list(PatternType)
        
        all_transactions = []
        
        for pattern in pattern_types:
            if pattern == PatternType.STRUCTURING:
                # Generate 2-5 structuring patterns with varying amounts
                for _ in range(random.randint(2, 5)):
                    amount = random.uniform(12000, 50000)
                    num_tx = random.randint(3, 8)
                    txs = self.simulate_structuring(amount, num_tx)
                    all_transactions.extend(txs)
                    
            elif pattern == PatternType.LAYERING:
                # Generate 1-3 layering patterns
                for _ in range(random.randint(1, 3)):
                    amount = random.uniform(50000, 200000)
                    num_layers = random.randint(3, 6)
                    txs = self.simulate_layering(amount, num_layers)
                    all_transactions.extend(txs)
                    
            elif pattern == PatternType.ROUND_TRIP:
                # Generate 2-4 round trip patterns
                for _ in range(random.randint(2, 4)):
                    amount = random.uniform(25000, 100000)
                    num_hops = random.randint(3, 5)
                    txs = self.simulate_round_trip(amount, num_hops)
                    all_transactions.extend(txs)
                    
            elif pattern == PatternType.RAPID_MOVEMENT:
                # Generate 1-2 rapid movement patterns
                for _ in range(random.randint(1, 2)):
                    amount = random.uniform(75000, 150000)
                    num_tx = random.randint(4, 7)
                    txs = self.simulate_rapid_movement(amount, num_tx)
                    all_transactions.extend(txs)
                    
            elif pattern == PatternType.FAN_IN:
                # Generate 2-3 fan-in patterns
                for _ in range(random.randint(2, 3)):
                    amount = random.uniform(100000, 300000)
                    num_senders = random.randint(5, 10)
                    txs = self.simulate_fan_in(amount, num_senders)
                    all_transactions.extend(txs)
                    
            elif pattern == PatternType.FAN_OUT:
                # Generate 2-3 fan-out patterns
                for _ in range(random.randint(2, 3)):
                    amount = random.uniform(100000, 300000)
                    num_receivers = random.randint(5, 10)
                    txs = self.simulate_fan_out(amount, num_receivers)
                    all_transactions.extend(txs)
                    
        # Sort transactions by timestamp
        all_transactions.sort(key=lambda x: x['timestamp'])
        return all_transactions

    def schedule_attack(self, pattern_type: PatternType, params: Dict, delay_minutes: int):
        """Schedule an attack to be executed after a delay"""
        execution_time = datetime.now() + timedelta(minutes=delay_minutes)
        self.scheduled_attacks.put((execution_time, pattern_type, params))
        print(f"\n{Fore.YELLOW}Scheduled {pattern_type.value} attack for {execution_time}{Style.RESET_ALL}")

    def start_scheduler(self):
        """Start the attack scheduler thread"""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        print(f"\n{Fore.GREEN}Attack scheduler started{Style.RESET_ALL}")

    def stop_scheduler(self):
        """Stop the attack scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        print(f"\n{Fore.RED}Attack scheduler stopped{Style.RESET_ALL}")

    def _scheduler_loop(self):
        """Main loop for executing scheduled attacks"""
        while self.running:
            now = datetime.now()
            
            # Check if there are attacks to execute
            while not self.scheduled_attacks.empty():
                execution_time, pattern_type, params = self.scheduled_attacks.queue[0]
                
                if execution_time <= now:
                    self.scheduled_attacks.get()  # Remove from queue
                    try:
                        self._execute_attack(pattern_type, params)
                    except Exception as e:
                        logging.error(f"Error executing scheduled attack: {str(e)}")
                else:
                    break
                    
            time.sleep(1)

    def _execute_attack(self, pattern_type: PatternType, params: Dict):
        """Execute a specific attack pattern"""
        try:
            if pattern_type == PatternType.STRUCTURING:
                self.simulate_structuring(params['amount'], params['num_tx'])
            elif pattern_type == PatternType.LAYERING:
                self.simulate_layering(params['amount'], params['num_layers'])
            elif pattern_type == PatternType.ROUND_TRIP:
                self.simulate_round_trip(params['amount'], params['num_hops'])
            elif pattern_type == PatternType.RAPID_MOVEMENT:
                self.simulate_rapid_movement(params['amount'], params['num_tx'])
            elif pattern_type == PatternType.FAN_IN:
                self.simulate_fan_in(params['amount'], params['num_senders'])
            elif pattern_type == PatternType.FAN_OUT:
                self.simulate_fan_out(params['amount'], params['num_receivers'])
            else:
                self.simulate_random_attack()
                
            print(f"\n{Fore.GREEN}Successfully executed scheduled {pattern_type.value} attack{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}Error executing {pattern_type.value} attack: {str(e)}{Style.RESET_ALL}")

def print_menu():
    """Print the attack simulation menu"""
    print(f"\n{Fore.CYAN}=== Money Laundering Pattern Simulator ==={Style.RESET_ALL}")
    print("1. Structuring Pattern")
    print("2. Layering Pattern")
    print("3. Round-Trip Pattern")
    print("4. Rapid Movement Pattern")
    print("5. Fan-In Pattern")
    print("6. Fan-Out Pattern")
    print("7. Random Pattern")
    print("8. Schedule an Attack")
    print("9. View Scheduled Attacks")
    print("10. Exit")
    print("\nEnter your choice (1-10): ")

def main():
    simulator = AttackSimulator()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    while True:
        print_menu()
        choice = input().strip()
        
        try:
            if choice == '1':
                amount = float(input("Enter total amount to structure: "))
                num_tx = int(input("Enter number of transactions: "))
                simulator.simulate_structuring(amount, num_tx)
                
            elif choice == '2':
                amount = float(input("Enter amount to layer: "))
                num_layers = int(input("Enter number of layers: "))
                simulator.simulate_layering(amount, num_layers)
                
            elif choice == '3':
                amount = float(input("Enter initial amount: "))
                num_hops = int(input("Enter number of hops: "))
                simulator.simulate_round_trip(amount, num_hops)
                
            elif choice == '4':
                amount = float(input("Enter amount to move: "))
                num_tx = int(input("Enter number of movements: "))
                simulator.simulate_rapid_movement(amount, num_tx)
                
            elif choice == '5':
                amount = float(input("Enter total amount for fan-in: "))
                num_senders = int(input("Enter number of senders: "))
                simulator.simulate_fan_in(amount, num_senders)
                
            elif choice == '6':
                amount = float(input("Enter total amount for fan-out: "))
                num_receivers = int(input("Enter number of receivers: "))
                simulator.simulate_fan_out(amount, num_receivers)
                
            elif choice == '7':
                simulator.simulate_random_attack()
                
            elif choice == '8':
                # Schedule an attack
                pattern = input("Enter pattern type to schedule (STRUCTURING, LAYERING, ROUND_TRIP, RAPID_MOVEMENT, FAN_IN, FAN_OUT): ")
                delay = int(input("Enter delay in minutes: "))
                
                if pattern == "STRUCTURING":
                    amount = float(input("Enter total amount to structure: "))
                    num_tx = int(input("Enter number of transactions: "))
                    simulator.schedule_attack(PatternType.STRUCTURING, {'amount': amount, 'num_tx': num_tx}, delay)
                    
                elif pattern == "LAYERING":
                    amount = float(input("Enter amount to layer: "))
                    num_layers = int(input("Enter number of layers: "))
                    simulator.schedule_attack(PatternType.LAYERING, {'amount': amount, 'num_layers': num_layers}, delay)
                    
                elif pattern == "ROUND_TRIP":
                    amount = float(input("Enter initial amount: "))
                    num_hops = int(input("Enter number of hops: "))
                    simulator.schedule_attack(PatternType.ROUND_TRIP, {'amount': amount, 'num_hops': num_hops}, delay)
                    
                elif pattern == "RAPID_MOVEMENT":
                    amount = float(input("Enter amount to move: "))
                    num_tx = int(input("Enter number of movements: "))
                    simulator.schedule_attack(PatternType.RAPID_MOVEMENT, {'amount': amount, 'num_tx': num_tx}, delay)
                    
                elif pattern == "FAN_IN":
                    amount = float(input("Enter total amount for fan-in: "))
                    num_senders = int(input("Enter number of senders: "))
                    simulator.schedule_attack(PatternType.FAN_IN, {'amount': amount, 'num_senders': num_senders}, delay)
                    
                elif pattern == "FAN_OUT":
                    amount = float(input("Enter total amount for fan-out: "))
                    num_receivers = int(input("Enter number of receivers: "))
                    simulator.schedule_attack(PatternType.FAN_OUT, {'amount': amount, 'num_receivers': num_receivers}, delay)
                    
                else:
                    print(f"{Fore.RED}Invalid pattern type.{Style.RESET_ALL}")
                
            elif choice == '9':
                # View scheduled attacks
                print(f"\n{Fore.CYAN}=== Scheduled Attacks ==={Style.RESET_ALL}")
                if simulator.scheduled_attacks.empty():
                    print("No scheduled attacks.")
                else:
                    for attack in list(simulator.scheduled_attacks.queue):
                        execution_time, pattern_type, params = attack
                        print(f"At {execution_time}, {pattern_type.value} with params {params}")
                
            elif choice == '10':
                print(f"\n{Fore.GREEN}Exiting simulator...{Style.RESET_ALL}")
                simulator.stop_scheduler()
                break
                
            else:
                print(f"{Fore.RED}Invalid choice. Please select 1-10.{Style.RESET_ALL}")
                
        except ValueError as e:
            print(f"{Fore.RED}Invalid input: {str(e)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
