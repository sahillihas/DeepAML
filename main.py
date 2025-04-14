"""
Money Laundering Detection System
--------------------------------
An end-to-end system that:
1. Ingests transaction data
2. Builds a knowledge graph
3. Detects suspicious patterns
4. Scores and reports potential money laundering cases
"""

import pandas as pd
import numpy as np
import datetime
import json
import networkx as nx
import matplotlib.pyplot as plt
from py2neo import Graph, Node, Relationship
from typing import List, Dict, Any, Tuple, Set
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AML_System")

class DataIngestion:
    """Handle ingestion of transaction data from various sources"""
    
    def __init__(self):
        self.transactions = []
        logger.info("Data Ingestion module initialized")
    
    def load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load transaction data from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                self.transactions = json.load(f)
            logger.info(f"Loaded {len(self.transactions)} transactions from {file_path}")
            return self.transactions
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
    
    def load_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Load transaction data from a pandas DataFrame"""
        self.transactions = df.to_dict(orient='records')
        logger.info(f"Loaded {len(self.transactions)} transactions from DataFrame")
        return self.transactions
    
    def load_from_json_string(self, json_string: str) -> List[Dict[str, Any]]:
        """Load transaction data from a JSON string"""
        try:
            self.transactions = json.loads(json_string)
            logger.info(f"Loaded {len(self.transactions)} transactions from JSON string")
            return self.transactions
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            return []
    
    def enrich_data(self) -> List[Dict[str, Any]]:
        """Enrich transaction data with additional information"""
        for tx in self.transactions:
            # Add day of week
            if 'timestamp' in tx:
                timestamp = datetime.datetime.fromisoformat(tx['timestamp'])
                tx['day_of_week'] = timestamp.strftime('%A')
                tx['hour_of_day'] = timestamp.hour
                
            # Add transaction country risk
            if 'sender_country' in tx and 'recipient_country' in tx:
                if tx['sender_country'] != tx['recipient_country']:
                    tx['cross_border'] = True
                    # Simple risk factor for demo - in reality would use country risk indices
                    high_risk_countries = {'CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG'}
                    if tx['sender_country'] in high_risk_countries or tx['recipient_country'] in high_risk_countries:
                        tx['country_risk'] = 'high'
                    else:
                        tx['country_risk'] = 'standard'
                else:
                    tx['cross_border'] = False
                    tx['country_risk'] = 'low'
        
        logger.info("Data enrichment completed")
        return self.transactions
    
    def get_transactions(self) -> List[Dict[str, Any]]:
        """Return the current transaction dataset"""
        return self.transactions


class KnowledgeGraphBuilder:
    """Build a knowledge graph from transaction data"""
    
    def __init__(self, use_neo4j=False, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        """
        Initialize the knowledge graph builder
        
        Args:
            use_neo4j: Whether to use Neo4j for storage (otherwise uses NetworkX)
            neo4j_uri: URI for Neo4j connection (e.g. "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.use_neo4j = use_neo4j
        
        if use_neo4j:
            try:
                self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
                logger.info("Connected to Neo4j database")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {str(e)}")
                self.use_neo4j = False
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()
            logger.info("Using NetworkX for graph representation")
    
    def build_graph(self, transactions: List[Dict[str, Any]]) -> Any:
        """
        Build a knowledge graph from transaction data
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            The graph object (either Neo4j Graph or NetworkX MultiDiGraph)
        """
        if self.use_neo4j:
            return self._build_neo4j_graph(transactions)
        else:
            return self._build_networkx_graph(transactions)
    
    def _build_networkx_graph(self, transactions: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """Build a NetworkX graph from transaction data"""
        # Clear existing graph
        self.graph = nx.MultiDiGraph()
        
        for tx in transactions:
            # Add sender account node
            sender_id = tx['sender_account']
            if not self.graph.has_node(sender_id):
                self.graph.add_node(sender_id, 
                                   type='account',
                                   name=tx['sender_name'],
                                   bank=tx['sender_bank'],
                                   country=tx['sender_country'])
            
            # Add recipient account node
            recipient_id = tx['recipient_account']
            if not self.graph.has_node(recipient_id):
                self.graph.add_node(recipient_id, 
                                   type='account',
                                   name=tx['recipient_name'],
                                   bank=tx['recipient_bank'],
                                   country=tx['recipient_country'])
            
            # Add transaction edge
            self.graph.add_edge(sender_id, recipient_id, 
                               transaction_id=tx['transaction_id'],
                               timestamp=tx['timestamp'],
                               amount=tx['amount'],
                               currency=tx['currency'],
                               reference=tx.get('reference', ''),
                               risk_score=tx.get('risk_score', 0))
        
        logger.info(f"Built NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _build_neo4j_graph(self, transactions: List[Dict[str, Any]]) -> Graph:
        """Build a Neo4j graph from transaction data"""
        # First clear database
        self.graph.run("MATCH (n) DETACH DELETE n")
        
        # Create unique constraints
        try:
            self.graph.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE")
        except Exception as e:
            logger.warning(f"Constraint creation error (may already exist): {str(e)}")
        
        # Create indices
        try:
            self.graph.run("CREATE INDEX account_country IF NOT EXISTS FOR (a:Account) ON (a.country)")
            self.graph.run("CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)")
        except Exception as e:
            logger.warning(f"Index creation error: {str(e)}")
        
        # Batch processing for better performance
        batch_size = 100
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i+batch_size]
            
            # Build Cypher query for batch insertion
            query = """
            UNWIND $transactions AS tx
            
            MERGE (sender:Account {account_id: tx.sender_account})
            ON CREATE SET 
                sender.name = tx.sender_name,
                sender.bank = tx.sender_bank,
                sender.country = tx.sender_country,
                sender.created_at = timestamp()
                
            MERGE (recipient:Account {account_id: tx.recipient_account})
            ON CREATE SET 
                recipient.name = tx.recipient_name,
                recipient.bank = tx.recipient_bank,
                recipient.country = tx.recipient_country,
                recipient.created_at = timestamp()
                
            CREATE (t:Transaction {transaction_id: tx.transaction_id})
            SET 
                t.timestamp = datetime(tx.timestamp),
                t.amount = tx.amount,
                t.currency = tx.currency,
                t.reference = tx.reference,
                t.risk_score = tx.risk_score,
                t.created_at = timestamp()
                
            CREATE (sender)-[:SENT]->(t)
            CREATE (t)-[:RECEIVED_BY]->(recipient)
            """
            
            self.graph.run(query, transactions=batch)
        
        # Count nodes and relationships
        node_count = self.graph.run("MATCH (n) RETURN count(n) AS count").data()[0]['count']
        rel_count = self.graph.run("MATCH ()-[r]->() RETURN count(r) AS count").data()[0]['count']
        
        logger.info(f"Built Neo4j graph with {node_count} nodes and {rel_count} relationships")
        return self.graph
    
    def visualize_graph(self, limit=50):
        """Visualize the graph (for NetworkX only)"""
        if self.use_neo4j:
            logger.warning("Graph visualization only available for NetworkX")
            return
        
        # Limit to a subset of nodes for visualization
        if self.graph.number_of_nodes() > limit:
            nodes = list(self.graph.nodes)[:limit]
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph)
        
        nx.draw_networkx_nodes(subgraph, pos, node_size=700)
        nx.draw_networkx_edges(subgraph, pos, arrowsize=20, width=2)
        nx.draw_networkx_labels(subgraph, pos, font_size=10)
        
        plt.axis('off')
        plt.title('Transaction Knowledge Graph')
        plt.tight_layout()
        plt.show()
    
    def find_paths(self, start_node, end_node, max_length=5):
        """Find paths between two nodes"""
        if self.use_neo4j:
            # Neo4j path finding
            query = """
            MATCH p = (:Account {account_id: $start})-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                    (:Account)-[:SENT*0..""" + str(max_length-1) + """]->(t:Transaction)-[:RECEIVED_BY]->
                    (:Account {account_id: $end})
            RETURN p, length(p) as path_length
            ORDER BY path_length
            LIMIT 10
            """
            result = self.graph.run(query, start=start_node, end=end_node).data()
            return result
        else:
            # NetworkX path finding
            try:
                paths = list(nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_length))
                return paths
            except nx.NetworkXNoPath:
                return []


class PatternDetection:
    """Detect suspicious patterns in the transaction knowledge graph"""
    
    def __init__(self, kg_builder):
        """
        Initialize pattern detection with a knowledge graph
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance
        """
        self.kg_builder = kg_builder
        self.use_neo4j = kg_builder.use_neo4j
        self.graph = kg_builder.graph
        self.patterns = {}
        logger.info("Pattern Detection module initialized")
    
    def detect_all_patterns(self) -> Dict[str, List[Dict]]:
        """Run all pattern detection methods and return results"""
        results = {}
        
        results['cycles'] = self.detect_cycles()
        results['structuring'] = self.detect_structuring()
        results['layering'] = self.detect_layering()
        results['fan_in'] = self.detect_fan_patterns(pattern_type='fan_in')
        results['fan_out'] = self.detect_fan_patterns(pattern_type='fan_out')
        results['rapid_movement'] = self.detect_rapid_movement()
        
        self.patterns = results
        return results
    
    def detect_cycles(self, max_length=6) -> List[Dict]:
        """
        Detect money moving in cycles through multiple entities
        
        Args:
            max_length: Maximum cycle length to consider
            
        Returns:
            List of detected cycle patterns
        """
        results = []
        
        if self.use_neo4j:
            # Neo4j cycle detection
            query = """
            MATCH p = (a:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                    (b:Account)-[:SENT*1..""" + str(max_length-2) + """]->(t:Transaction)-[:RECEIVED_BY]->(a)
            WHERE a <> b
            WITH p, relationships(p) AS rels
            WITH p, 
                 [r IN rels | r.amount] AS amounts,
                 a.account_id AS start_account,
                 a.name AS start_name
            WITH p, 
                 start_account,
                 start_name,
                 amounts,
                 reduce(total = 0, x IN amounts | total + x) AS total_volume,
                 size(apoc.coll.toSet([r IN relationships(p) | startNode(r).country])) AS country_count
            WHERE country_count >= 2
            RETURN start_account, 
                   start_name,
                   total_volume,
                   country_count,
                   size(amounts) AS cycle_length,
                   startNode(relationships(p)[0]).country AS start_country
            ORDER BY total_volume DESC
            LIMIT 20
            """
            
            try:
                result = self.graph.run(query).data()
                for record in result:
                    results.append({
                        'pattern_type': 'cycle',
                        'start_account': record['start_account'],
                        'start_name': record['start_name'],
                        'total_volume': record['total_volume'],
                        'cycle_length': record['cycle_length'],
                        'country_count': record['country_count'],
                        'risk_score': self._calculate_pattern_risk('cycle', record)
                    })
            except Exception as e:
                logger.error(f"Error in cycle detection (Neo4j): {str(e)}")
        
        else:
            # NetworkX cycle detection
            try:
                # Find simple cycles in the graph
                cycles = list(nx.simple_cycles(self.graph))
                cycles = [c for c in cycles if 2 < len(c) <= max_length]
                
                for cycle in cycles:
                    # Skip if cycle is too small
                    if len(cycle) <= 2:
                        continue
                        
                    # Get total volume and country count
                    total_volume = 0
                    countries = set()
                    
                    for i in range(len(cycle)):
                        sender = cycle[i]
                        recipient = cycle[(i+1) % len(cycle)]
                        
                        for _, _, data in self.graph.edges(sender, recipient, data=True):
                            total_volume += data.get('amount', 0)
                            
                        sender_data = self.graph.nodes[sender]
                        countries.add(sender_data.get('country', ''))
                    
                    # Only include multi-country cycles with significant volume
                    if len(countries) >= 2:
                        start_account = cycle[0]
                        start_data = self.graph.nodes[start_account]
                        
                        results.append({
                            'pattern_type': 'cycle',
                            'start_account': start_account,
                            'start_name': start_data.get('name', 'Unknown'),
                            'total_volume': total_volume,
                            'cycle_length': len(cycle),
                            'country_count': len(countries),
                            'risk_score': self._calculate_pattern_risk('cycle', {
                                'total_volume': total_volume,
                                'cycle_length': len(cycle),
                                'country_count': len(countries)
                            })
                        })
            
            except Exception as e:
                logger.error(f"Error in cycle detection (NetworkX): {str(e)}")
        
        logger.info(f"Detected {len(results)} cycle patterns")
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)
    
    def detect_structuring(self, threshold=100000, max_days=5) -> List[Dict]:
        """
        Detect structuring (breaking large sums into smaller transactions)
        
        Args:
            threshold: Amount threshold to consider for structuring
            max_days: Maximum days between first and last transaction
            
        Returns:
            List of detected structuring patterns
        """
        results = []
        
        if self.use_neo4j:
            # Neo4j structuring detection
            query = """
            MATCH (a:Account)-[:SENT]->(t1:Transaction)-[:RECEIVED_BY]->(b:Account)
            WITH a, b, collect(t1) AS transactions
            WHERE size(transactions) >= 3
            WITH a, b, transactions,
                 [t IN transactions | t.amount] AS amounts,
                 [t IN transactions | t.timestamp] AS timestamps
            WITH a, b, transactions, amounts,
                 min(timestamps) AS first_tx,
                 max(timestamps) AS last_tx,
                 sum(amounts) AS total_amount,
                 size(amounts) AS tx_count
            WHERE total_amount >= $threshold
            AND duration.inDays(first_tx, last_tx).days <= $max_days
            AND max(amounts) < (total_amount * 0.5)
            RETURN a.account_id AS sender_account,
                   a.name AS sender_name,
                   b.account_id AS recipient_account,
                   b.name AS recipient_name,
                   total_amount,
                   tx_count,
                   duration.inDays(first_tx, last_tx).days AS days_span
            ORDER BY total_amount DESC
            LIMIT 20
            """
            
            try:
                result = self.graph.run(query, threshold=threshold, max_days=max_days).data()
                for record in result:
                    results.append({
                        'pattern_type': 'structuring',
                        'sender_account': record['sender_account'],
                        'sender_name': record['sender_name'],
                        'recipient_account': record['recipient_account'],
                        'recipient_name': record['recipient_name'],
                        'total_amount': record['total_amount'],
                        'transaction_count': record['tx_count'],
                        'days_span': record['days_span'],
                        'risk_score': self._calculate_pattern_risk('structuring', record)
                    })
            except Exception as e:
                logger.error(f"Error in structuring detection (Neo4j): {str(e)}")
        
        else:
            # NetworkX structuring detection
            account_pairs = {}
            
            # Group transactions by sender-recipient pairs
            for u, v, data in self.graph.edges(data=True):
                if 'amount' in data and 'timestamp' in data:
                    pair_key = f"{u}|{v}"
                    if pair_key not in account_pairs:
                        account_pairs[pair_key] = {
                            'sender': u,
                            'recipient': v,
                            'transactions': []
                        }
                    
                    account_pairs[pair_key]['transactions'].append({
                        'amount': data['amount'],
                        'timestamp': datetime.datetime.fromisoformat(data['timestamp'])
                    })
            
            # Analyze each account pair for structuring
            for pair_key, pair_data in account_pairs.items():
                txs = pair_data['transactions']
                
                # Skip if too few transactions
                if len(txs) < 3:
                    continue
                
                # Calculate metrics
                amounts = [tx['amount'] for tx in txs]
                timestamps = [tx['timestamp'] for tx in txs]
                total_amount = sum(amounts)
                max_amount = max(amounts)
                
                first_tx = min(timestamps)
                last_tx = max(timestamps)
                days_span = (last_tx - first_tx).days
                
                # Check for structuring pattern
                if (total_amount >= threshold and 
                    days_span <= max_days and 
                    max_amount < (total_amount * 0.5)):
                    
                    sender_data = self.graph.nodes[pair_data['sender']]
                    recipient_data = self.graph.nodes[pair_data['recipient']]
                    
                    results.append({
                        'pattern_type': 'structuring',
                        'sender_account': pair_data['sender'],
                        'sender_name': sender_data.get('name', 'Unknown'),
                        'recipient_account': pair_data['recipient'],
                        'recipient_name': recipient_data.get('name', 'Unknown'),
                        'total_amount': total_amount,
                        'transaction_count': len(txs),
                        'days_span': days_span,
                        'risk_score': self._calculate_pattern_risk('structuring', {
                            'total_amount': total_amount,
                            'tx_count': len(txs),
                            'days_span': days_span
                        })
                    })
        
        logger.info(f"Detected {len(results)} structuring patterns")
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)
    
    def detect_layering(self, min_path_length=3, min_countries=3) -> List[Dict]:
        """
        Detect layering chains (multiple hops to obscure source and destination)
        
        Args:
            min_path_length: Minimum path length to consider as layering
            min_countries: Minimum number of countries in the path
            
        Returns:
            List of detected layering patterns
        """
        results = []
        
        if self.use_neo4j:
            # Neo4j layering detection
            query = """
            MATCH path = (a:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                        (b:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                        (c:Account)
            WHERE a <> c
            WITH path, relationships(path) AS rels,
                 a.account_id AS source_account,
                 a.name AS source_name,
                 c.account_id AS dest_account,
                 c.name AS dest_name
            WITH path, rels, source_account, source_name, dest_account, dest_name,
                 [r IN rels | startNode(r).country] + [endNode(last(rels)).country] AS countries,
                 [r IN rels | r.amount] AS amounts
            WITH path, source_account, source_name, dest_account, dest_name,
                 size(apoc.coll.toSet(countries)) AS country_count,
                 amounts,
                 min(amounts) AS min_amount,
                 max(amounts) AS max_amount,
                 size(rels) AS path_length
            WHERE country_count >= $min_countries
            AND path_length >= $min_path_length
            AND abs(1 - (max_amount / min_amount)) <= 0.2
            RETURN source_account, source_name, dest_account, dest_name,
                   path_length, country_count, min_amount, max_amount,
                   min_amount / max_amount AS amount_ratio
            ORDER BY path_length DESC, country_count DESC
            LIMIT 20
            """
            
            try:
                result = self.graph.run(query, min_path_length=min_path_length, 
                                       min_countries=min_countries).data()
                for record in result:
                    results.append({
                        'pattern_type': 'layering',
                        'source_account': record['source_account'],
                        'source_name': record['source_name'],
                        'destination_account': record['dest_account'],
                        'destination_name': record['dest_name'],
                        'path_length': record['path_length'],
                        'country_count': record['country_count'],
                        'amount_consistency': record['amount_ratio'],
                        'average_amount': (record['min_amount'] + record['max_amount']) / 2,
                        'risk_score': self._calculate_pattern_risk('layering', record)
                    })
            except Exception as e:
                logger.error(f"Error in layering detection (Neo4j): {str(e)}")
        
        else:
            # NetworkX layering detection - we'll need to find all paths and analyze them
            # This is a simplified approach due to complexity
            account_nodes = [n for n, attr in self.graph.nodes(data=True) 
                            if attr.get('type', '') == 'account']
            
            # Limit the number of paths to avoid excessive computation
            max_pairs_to_check = 100
            checked_pairs = 0
            
            for i, source in enumerate(account_nodes):
                for dest in account_nodes[i+1:]:
                    if checked_pairs >= max_pairs_to_check:
                        break
                    
                    # Find paths between source and destination
                    try:
                        paths = list(nx.all_simple_paths(self.graph, source=source, 
                                                       target=dest, cutoff=min_path_length+3))
                        
                        for path in paths:
                            if len(path) < min_path_length:
                                continue
                                
                            # Collect countries and amounts
                            countries = set()
                            amounts = []
                            
                            for i in range(len(path)-1):
                                sender = path[i]
                                recipient = path[i+1]
                                
                                # Add sender country
                                sender_data = self.graph.nodes[sender]
                                if 'country' in sender_data:
                                    countries.add(sender_data['country'])
                                
                                # Add amount if this is a valid edge
                                for _, _, data in self.graph.edges(sender, recipient, data=True):
                                    if 'amount' in data:
                                        amounts.append(data['amount'])
                            
                            # Add recipient country (last node)
                            recipient_data = self.graph.nodes[path[-1]]
                            if 'country' in recipient_data:
                                countries.add(recipient_data['country'])
                            
                            # Check if this qualifies as layering
                            if (len(countries) >= min_countries and len(amounts) >= min_path_length-1):
                                min_amount = min(amounts)
                                max_amount = max(amounts)
                                amount_ratio = min_amount / max_amount if max_amount > 0 else 0
                                
                                # Check for consistent amounts (80% similarity)
                                if amount_ratio >= 0.8:
                                    source_data = self.graph.nodes[source]
                                    dest_data = self.graph.nodes[dest]
                                    
                                    results.append({
                                        'pattern_type': 'layering',
                                        'source_account': source,
                                        'source_name': source_data.get('name', 'Unknown'),
                                        'destination_account': dest,
                                        'destination_name': dest_data.get('name', 'Unknown'),
                                        'path_length': len(path),
                                        'country_count': len(countries),
                                        'amount_consistency': amount_ratio,
                                        'average_amount': sum(amounts) / len(amounts),
                                        'risk_score': self._calculate_pattern_risk('layering', {
                                            'path_length': len(path),
                                            'country_count': len(countries),
                                            'amount_ratio': amount_ratio,
                                            'min_amount': min_amount,
                                            'max_amount': max_amount
                                        })
                                    })
                    except Exception as e:
                        pass  # Ignore path finding errors
                    
                    checked_pairs += 1
        
        logger.info(f"Detected {len(results)} layering patterns")
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)
    
    def detect_fan_patterns(self, pattern_type='fan_in', min_connections=3) -> List[Dict]:
        """
        Detect fan-in or fan-out patterns (multiple sources to one destination or vice versa)
        
        Args:
            pattern_type: 'fan_in' or 'fan_out'
            min_connections: Minimum number of connections to consider as a fan pattern
            
        Returns:
            List of detected fan patterns
        """
        results = []
        
        if self.use_neo4j:
            # Neo4j fan pattern detection
            if pattern_type == 'fan_in':
                query = """
                MATCH (source:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(dest:Account)
                WITH dest, count(source) AS source_count, collect(source) AS sources
                WHERE source_count >= $min_connections
                WITH dest, source_count, sources,
                     [s IN sources | s.country] AS countries
                WITH dest, source_count,
                     size(apoc.coll.toSet(countries)) AS country_count
                WHERE country_count >= 2
                RETURN dest.account_id AS account_id,
                       dest
                       source.name AS account_name,
                       dest_count,
                       country_count
                ORDER BY dest_count DESC
                LIMIT 15
                """
            
            try:
                result = self.graph.run(query, min_connections=min_connections).data()
                for record in result:
                    results.append({
                        'pattern_type': pattern_type,
                        'account_id': record['account_id'],
                        'account_name': record['account_name'],
                        'connection_count': record['source_count'] if pattern_type == 'fan_in' else record['dest_count'],
                        'country_count': record['country_count'],
                        'risk_score': self._calculate_pattern_risk(pattern_type, record)
                    })
            except Exception as e:
                logger.error(f"Error in {pattern_type} detection (Neo4j): {str(e)}")
        
        else:
            # NetworkX fan pattern detection
            if pattern_type == 'fan_in':
                # Count incoming edges for each node
                for node in self.graph.nodes():
                    incoming = list(self.graph.predecessors(node))
                    
                    if len(incoming) >= min_connections:
                        # Get countries of sources
                        countries = set()
                        for source in incoming:
                            source_data = self.graph.nodes[source]
                            if 'country' in source_data:
                                countries.add(source_data['country'])
                        
                        if len(countries) >= 2:
                            node_data = self.graph.nodes[node]
                            results.append({
                                'pattern_type': 'fan_in',
                                'account_id': node,
                                'account_name': node_data.get('name', 'Unknown'),
                                'connection_count': len(incoming),
                                'country_count': len(countries),
                                'risk_score': self._calculate_pattern_risk('fan_in', {
                                    'source_count': len(incoming),
                                    'country_count': len(countries)
                                })
                            })
            else:  # fan_out
                # Count outgoing edges for each node
                for node in self.graph.nodes():
                    outgoing = list(self.graph.successors(node))
                    
                    if len(outgoing) >= min_connections:
                        # Get countries of destinations
                        countries = set()
                        for dest in outgoing:
                            dest_data = self.graph.nodes[dest]
                            if 'country' in dest_data:
                                countries.add(dest_data['country'])
                        
                        if len(countries) >= 2:
                            node_data = self.graph.nodes[node]
                            results.append({
                                'pattern_type': 'fan_out',
                                'account_id': node,
                                'account_name': node_data.get('name', 'Unknown'),
                                'connection_count': len(outgoing),
                                'country_count': len(countries),
                                'risk_score': self._calculate_pattern_risk('fan_out', {
                                    'dest_count': len(outgoing),
                                    'country_count': len(countries)
                                })
                            })
        
        logger.info(f"Detected {len(results)} {pattern_type} patterns")
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)
    
    def detect_rapid_movement(self, max_hours=48, min_links=3) -> List[Dict]:
        """
        Detect rapid movement of money through multiple entities in a short timeframe
        
        Args:
            max_hours: Maximum hours between first and last transaction
            min_links: Minimum number of links in the chain
            
        Returns:
            List of detected rapid movement patterns
        """
        results = []
        
        if self.use_neo4j:
            # Neo4j rapid movement detection
            query = """
            MATCH path = (start:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                        (:Account)-[:SENT*1..""" + str(min_links) + """]->(t:Transaction)-[:RECEIVED_BY]->(end:Account)
            WHERE start <> end
            WITH path, relationships(path) AS rels,
                 start.account_id AS start_account,
                 start.name AS start_name,
                 end.account_id AS end_account,
                 end.name AS end_name
            WITH path, rels, start_account, start_name, end_account, end_name,
                 [r IN rels | r.timestamp] AS timestamps,
                 [r IN rels | r.amount] AS amounts
            WITH start_account, start_name, end_account, end_name,
                 min(timestamps) AS first_tx,
                 max(timestamps) AS last_tx,
                 duration.inSeconds(min(timestamps), max(timestamps))/3600.0 AS hours_elapsed,
                 size(rels) AS chain_length,
                 reduce(s = 0, a IN amounts | s + a) AS total_volume
            WHERE hours_elapsed <= $max_hours
            AND chain_length >= $min_links
            RETURN start_account, start_name, end_account, end_name,
                   chain_length, hours_elapsed, total_volume
            ORDER BY hours_elapsed ASC
            LIMIT 20
            """
            
            try:
                result = self.graph.run(query, max_hours=max_hours, min_links=min_links).data()
                for record in result:
                    results.append({
                        'pattern_type': 'rapid_movement',
                        'start_account': record['start_account'],
                        'start_name': record['start_name'],
                        'end_account': record['end_account'],
                        'end_name': record['end_name'],
                        'chain_length': record['chain_length'],
                        'hours_elapsed': record['hours_elapsed'],
                        'total_volume': record['total_volume'],
                        'risk_score': self._calculate_pattern_risk('rapid_movement', record)
                    })
            except Exception as e:
                logger.error(f"Error in rapid movement detection (Neo4j): {str(e)}")
        
        else:
            # NetworkX rapid movement detection
            # This is a simplified version as complete path analysis is computationally expensive
            
            # First, get all account nodes
            account_nodes = [n for n, attr in self.graph.nodes(data=True) 
                           if attr.get('type', '') == 'account']
            
            # Limit the number of paths to analyze
            max_pairs = 100
            checked = 0
            
            for i, start in enumerate(account_nodes):
                for end in account_nodes[i+1:]:
                    if checked >= max_pairs:
                        break
                        
                    if start == end:
                        continue
                    
                    try:
                        # Find paths with specified minimum length
                        paths = list(nx.all_simple_paths(self.graph, source=start, 
                                                      target=end, cutoff=min_links+3))
                        
                        for path in paths:
                            if len(path) < min_links:
                                continue
                                
                            # Get timestamps and amounts
                            timestamps = []
                            amounts = []
                            
                            for i in range(len(path)-1):
                                sender = path[i]
                                recipient = path[i+1]
                                
                                for _, _, data in self.graph.edges(sender, recipient, data=True):
                                    if 'timestamp' in data and 'amount' in data:
                                        timestamps.append(datetime.datetime.fromisoformat(data['timestamp']))
                                        amounts.append(data['amount'])
                            
                            if not timestamps or len(timestamps) < min_links:
                                continue
                                
                            # Calculate time elapsed
                            first_tx = min(timestamps)
                            last_tx = max(timestamps)
                            hours_elapsed = (last_tx - first_tx).total_seconds() / 3600
                            
                            if hours_elapsed <= max_hours:
                                start_data = self.graph.nodes[start]
                                end_data = self.graph.nodes[end]
                                
                                results.append({
                                    'pattern_type': 'rapid_movement',
                                    'start_account': start,
                                    'start_name': start_data.get('name', 'Unknown'),
                                    'end_account': end,
                                    'end_name': end_data.get('name', 'Unknown'),
                                    'chain_length': len(path),
                                    'hours_elapsed': hours_elapsed,
                                    'total_volume': sum(amounts),
                                    'risk_score': self._calculate_pattern_risk('rapid_movement', {
                                        'chain_length': len(path),
                                        'hours_elapsed': hours_elapsed,
                                        'total_volume': sum(amounts)
                                    })
                                })
                    except Exception:
                        pass  # Ignore errors in path finding
                        
                    checked += 1
        
        logger.info(f"Detected {len(results)} rapid movement patterns")
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)
    
    def _calculate_pattern_risk(self, pattern_type, data) -> float:
        """Calculate risk score for a detected pattern"""
        if pattern_type == 'cycle':
            # Higher risk for cycles with more countries and larger volumes
            base_score = 0.7
            volume_factor = min(1.0, data.get('total_volume', 0) / 1000000) * 0.15
            country_factor = min(1.0, data.get('country_count', 0) / 5) * 0.15
            return base_score + volume_factor + country_factor
            
        elif pattern_type == 'structuring':
            # Higher risk for structuring with more transactions and larger amounts
            base_score = 0.65
            volume_factor = min(1.0, data.get('total_amount', 0) / 500000) * 0.2
            tx_factor = min(1.0, data.get('tx_count', 0) / 10) * 0.15
            return base_score + volume_factor + tx_factor
            
        elif pattern_type == 'layering':
            # Higher risk for layering with more countries and longer paths
            base_score = 0.75
            path_factor = min(1.0, data.get('path_length', 0) / 6) * 0.1
            country_factor = min(1.0, data.get('country_count', 0) / 4) * 0.15
            return base_score + path_factor + country_factor
            
        elif pattern_type in ('fan_in', 'fan_out'):
            # Higher risk for fan patterns with more connections and countries
            base_score = 0.6
            connection_factor = min(1.0, data.get('connection_count', 0) / 10) * 0.2
            country_factor = min(1.0, data.get('country_count', 0) / 4) * 0.2
            return base_score + connection_factor + country_factor
            
        elif pattern_type == 'rapid_movement':
            # Higher risk for rapid movement with more links and shorter timeframes
            base_score = 0.7
            chain_factor = min(1.0, data.get('chain_length', 0) / 5) * 0.1
            time_factor = (1.0 - min(1.0, data.get('hours_elapsed', 48) / 48)) * 0.2
            return base_score + chain_factor + time_factor
            
        else:
            return 0.5  # Default risk score


class RiskScoring:
    """Calculate risk scores for entities and transactions"""
    
    def __init__(self, kg_builder, pattern_detector):
        """
        Initialize risk scoring
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance
            pattern_detector: PatternDetection instance
        """
        self.kg_builder = kg_builder
        self.pattern_detector = pattern_detector
        self.use_neo4j = kg_builder.use_neo4j
        self.graph = kg_builder.graph
        self.entity_risks = {}
        logger.info("Risk Scoring module initialized")
    
    def score_all_entities(self) -> Dict[str, float]:
        """Calculate risk scores for all entities in the graph"""
        if self.use_neo4j:
            return self._score_entities_neo4j()
        else:
            return self._score_entities_networkx()
    
    def _score_entities_neo4j(self) -> Dict[str, float]:
        """Calculate risk scores for entities using Neo4j"""
        entity_risks = {}
        
        # Basic query to retrieve all accounts with their properties
        query = """
        MATCH (a:Account)
        RETURN a.account_id AS account_id,
               a.name AS name,
               a.country AS country
        """
        
        accounts = self.graph.run(query).data()
        
        # Calculate country risk factor
        high_risk_countries = {'CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG'}
        medium_risk_countries = {'CH', 'HK', 'SG', 'MO', 'BS', 'AI'}
        
        # Get pattern involvement
        pattern_data = self.pattern_detector.patterns
        
        for account in accounts:
            account_id = account['account_id']
            
            # Base risk based on country
            base_risk = 0.3  # Default base risk
            
            if account['country'] in high_risk_countries:
                base_risk = 0.5
            elif account['country'] in medium_risk_countries:
                base_risk = 0.4
            
            # Increase risk based on pattern involvement
            pattern_risk = 0.0
            pattern_count = 0
            
            # Check all patterns for this account
            for pattern_type, patterns in pattern_data.items():
                for pattern in patterns:
                    # Check if this account is involved in the pattern
                    account_match = False
                    
                    if pattern_type == 'cycle' and pattern.get('start_account') == account_id:
                        account_match = True
                    elif pattern_type == 'structuring' and (pattern.get('sender_account') == account_id or 
                                                       pattern.get('recipient_account') == account_id):
                        account_match = True
                    elif pattern_type == 'layering' and (pattern.get('source_account') == account_id or 
                                                    pattern.get('destination_account') == account_id):
                        account_match = True
                    elif pattern_type in ('fan_in', 'fan_out') and pattern.get('account_id') == account_id:
                        account_match = True
                    elif pattern_type == 'rapid_movement' and (pattern.get('start_account') == account_id or 
                                                          pattern.get('end_account') == account_id):
                        account_match = True
                    
                    if account_match:
                        pattern_count += 1
                        pattern_risk += pattern.get('risk_score', 0)
            
            # Calculate final risk score
            if pattern_count > 0:
                avg_pattern_risk = pattern_risk / pattern_count
                final_risk = base_risk * 0.4 + avg_pattern_risk * 0.6
            else:
                final_risk = base_risk
                
            # Cap risk at 1.0
            final_risk = min(1.0, final_risk)
            
            # Store calculated risk
            entity_risks[account_id] = {
                'account_id': account_id,
                'name': account['name'],
                'country': account['country'],
                'base_risk': base_risk,
                'pattern_count': pattern_count,
                'final_risk_score': final_risk
            }
            
            # Update risk in database
            update_query = """
            MATCH (a:Account {account_id: $account_id})
            SET a.risk_score = $risk_score
            """
            self.graph.run(update_query, account_id=account_id, risk_score=final_risk)
        
        logger.info(f"Calculated risk scores for {len(entity_risks)} entities")
        self.entity_risks = entity_risks
        return entity_risks
    
    def _score_entities_networkx(self) -> Dict[str, float]:
        """Calculate risk scores for entities using NetworkX"""
        entity_risks = {}
        
        # Get all account nodes
        accounts = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type', '') == 'account':
                accounts.append({
                    'account_id': node,
                    'name': data.get('name', 'Unknown'),
                    'country': data.get('country', '')
                })
        
        # Calculate country risk factor
        high_risk_countries = {'CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG'}
        medium_risk_countries = {'CH', 'HK', 'SG', 'MO', 'BS', 'AI'}
        
        # Get pattern involvement
        pattern_data = self.pattern_detector.patterns
        
        for account in accounts:
            account_id = account['account_id']
            
            # Base risk based on country
            base_risk = 0.3  # Default base risk
            
            if account['country'] in high_risk_countries:
                base_risk = 0.5
            elif account['country'] in medium_risk_countries:
                base_risk = 0.4
            
            # Increase risk based on pattern involvement
            pattern_risk = 0.0
            pattern_count = 0
            
            # Check all patterns for this account
            for pattern_type, patterns in pattern_data.items():
                for pattern in patterns:
                    # Check if this account is involved in the pattern
                    account_match = False
                    
                    if pattern_type == 'cycle' and pattern.get('start_account') == account_id:
                        account_match = True
                    elif pattern_type == 'structuring' and (pattern.get('sender_account') == account_id or 
                                                       pattern.get('recipient_account') == account_id):
                        account_match = True
                    elif pattern_type == 'layering' and (pattern.get('source_account') == account_id or 
                                                    pattern.get('destination_account') == account_id):
                        account_match = True
                    elif pattern_type in ('fan_in', 'fan_out') and pattern.get('account_id') == account_id:
                        account_match = True
                    elif pattern_type == 'rapid_movement' and (pattern.get('start_account') == account_id or 
                                                          pattern.get('end_account') == account_id):
                        account_match = True
                    
                    if account_match:
                        pattern_count += 1
                        pattern_risk += pattern.get('risk_score', 0)
            
            # Calculate final risk score
            if pattern_count > 0:
                avg_pattern_risk = pattern_risk / pattern_count
                final_risk = base_risk * 0.4 + avg_pattern_risk * 0.6
            else:
                final_risk = base_risk
                
            # Cap risk at 1.0
            final_risk = min(1.0, final_risk)
            
            # Store calculated risk
            entity_risks[account_id] = {
                'account_id': account_id,
                'name': account['name'],
                'country': account['country'],
                'base_risk': base_risk,
                'pattern_count': pattern_count,
                'final_risk_score': final_risk
            }
            
            # Update risk in graph
            self.graph.nodes[account_id]['risk_score'] = final_risk
        
        logger.info(f"Calculated risk scores for {len(entity_risks)} entities")
        self.entity_risks = entity_risks
        return entity_risks
    
    def get_highest_risk_entities(self, top_n=10) -> List[Dict[str, Any]]:
        """Get the highest risk entities"""
        entities = list(self.entity_risks.values())
        return sorted(entities, key=lambda x: x['final_risk_score'], reverse=True)[:top_n]


class CaseManagement:
    """Generate and manage alerts for investigation"""
    
    def __init__(self, risk_scorer, pattern_detector):
        """
        Initialize case management
        
        Args:
            risk_scorer: RiskScoring instance
            pattern_detector: PatternDetection instance
        """
        self.risk_scorer = risk_scorer
        self.pattern_detector = pattern_detector
        self.cases = []
        logger.info("Case Management module initialized")
    
    def generate_cases(self, risk_threshold=0.75) -> List[Dict[str, Any]]:
        """
        Generate cases for investigation based on risk scores and patterns
        
        Args:
            risk_threshold: Minimum risk score to generate a case
            
        Returns:
            List of cases for investigation
        """
        cases = []
        case_id = 1
        
        # Generate cases for high-risk entities
        high_risk_entities = self.risk_scorer.get_highest_risk_entities(top_n=20)
        
        for entity in high_risk_entities:
            if entity['final_risk_score'] >= risk_threshold:
                # Create a case for this entity
                case = {
                    'case_id': f"CASE-{case_id:04d}",
                    'priority': 'High' if entity['final_risk_score'] > 0.85 else 'Medium',
                    'entity_id': entity['account_id'],
                    'entity_name': entity['name'],
                    'risk_score': entity['final_risk_score'],
                    'involved_patterns': self._get_entity_patterns(entity['account_id']),
                    'detection_timestamp': datetime.datetime.now().isoformat(),
                    'status': 'Open',
                    'case_type': 'High-Risk Entity'
                }
                cases.append(case)
                case_id += 1
        
        # Generate cases for specific high-risk patterns
        for pattern_type, patterns in self.pattern_detector.patterns.items():
            # Focus on the most risky patterns
            top_patterns = sorted(patterns, key=lambda x: x.get('risk_score', 0), reverse=True)[:5]
            
            for pattern in top_patterns:
                if pattern.get('risk_score', 0) >= risk_threshold:
                    # Create a case for this pattern
                    case = {
                        'case_id': f"CASE-{case_id:04d}",
                        'priority': 'High' if pattern.get('risk_score', 0) > 0.85 else 'Medium',
                        'pattern_type': pattern_type,
                        'risk_score': pattern.get('risk_score', 0),
                        'detection_timestamp': datetime.datetime.now().isoformat(),
                        'status': 'Open',
                        'case_type': 'Suspicious Pattern',
                        'pattern_details': pattern
                    }
                    cases.append(case)
                    case_id += 1
        
        logger.info(f"Generated {len(cases)} cases for investigation")
        self.cases = cases
        return cases
    
    def _get_entity_patterns(self, account_id: str) -> List[Dict[str, Any]]:
        """Get patterns involving an entity"""
        involved_patterns = []
        
        for pattern_type, patterns in self.pattern_detector.patterns.items():
            for pattern in patterns:
                # Check if this account is involved in the pattern
                account_match = False
                
                if pattern_type == 'cycle' and pattern.get('start_account') == account_id:
                    account_match = True
                elif pattern_type == 'structuring' and (pattern.get('sender_account') == account_id or 
                                                   pattern.get('recipient_account') == account_id):
                    account_match = True
                elif pattern_type == 'layering' and (pattern.get('source_account') == account_id or 
                                                pattern.get('destination_account') == account_id):
                    account_match = True
                elif pattern_type in ('fan_in', 'fan_out') and pattern.get('account_id') == account_id:
                    account_match = True
                elif pattern_type == 'rapid_movement' and (pattern.get('start_account') == account_id or 
                                                      pattern.get('end_account') == account_id):
                    account_match = True
                
                if account_match:
                    involved_patterns.append({
                        'pattern_type': pattern_type,
                        'risk_score': pattern.get('risk_score', 0),
                        'details': pattern
                    })
        
        return sorted(involved_patterns, key=lambda x: x['risk_score'], reverse=True)
    
    def get_case_details(self, case_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific case"""
        for case in self.cases:
            if case['case_id'] == case_id:
                return case
        return None
    
    def update_case_status(self, case_id: str, new_status: str) -> bool:
        """Update the status of a case"""
        for case in self.cases:
            if case['case_id'] == case_id:
                case['status'] = new_status
                case['last_updated'] = datetime.datetime.now().isoformat()
                logger.info(f"Updated case {case_id} status to {new_status}")
                return True
        logger.warning(f"Case {case_id} not found")
        return False


class ReportGenerator:
    """Generate reports and visualizations from detection results"""
    
    def __init__(self, case_manager, risk_scorer, pattern_detector):
        """
        Initialize report generator
        
        Args:
            case_manager: CaseManagement instance
            risk_scorer: RiskScoring instance
            pattern_detector: PatternDetection instance
        """
        self.case_manager = case_manager
        self.risk_scorer = risk_scorer
        self.pattern_detector = pattern_detector
        logger.info("Report Generator initialized")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of detection results"""
        # Count patterns by type
        pattern_counts = {}
        for pattern_type, patterns in self.pattern_detector.patterns.items():
            pattern_counts[pattern_type] = len(patterns)
        
        # Calculate risk distribution
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
        for entity_risk in self.risk_scorer.entity_risks.values():
            score = entity_risk['final_risk_score']
            if score < 0.5:
                risk_distribution['low'] += 1
            elif score < 0.75:
                risk_distribution['medium'] += 1
            else:
                risk_distribution['high'] += 1
        
        # Case status summary
        case_status = {'Open': 0, 'In Progress': 0, 'Closed': 0}
        for case in self.case_manager.cases:
            status = case['status']
            if status in case_status:
                case_status[status] += 1
        
        # Highest risk entities
        high_risk_entities = self.risk_scorer.get_highest_risk_entities(top_n=5)
        
        # Generate summary report
        report = {
            'report_id': f"REP-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'detection_summary': {
                'total_patterns_detected': sum(pattern_counts.values()),
                'pattern_distribution': pattern_counts,
                'risk_distribution': risk_distribution,
                'case_status': case_status
            },
            'high_risk_entities': high_risk_entities,
            'top_cases': self.case_manager.cases[:5]
        }
        
        logger.info("Generated summary report")
        return report
    
    def generate_suspicious_activity_report(self, case_id: str) -> Dict[str, Any]:
        """Generate a suspicious activity report (SAR) for a specific case"""
        case = self.case_manager.get_case_details(case_id)
        
        if not case:
            logger.warning(f"Case {case_id} not found")
            return None
        
        # Generate SAR content based on case type
        if case['case_type'] == 'High-Risk Entity':
            entity_id = case['entity_id']
            entity_name = case['entity_name']
            entity_risk = self.risk_scorer.entity_risks.get(entity_id, {})
            involved_patterns = case['involved_patterns']
            
            narrative = (f"Entity {entity_name} (ID: {entity_id}) has been identified as a high-risk entity "
                        f"with a risk score of {case['risk_score']:.2f}. "
                        f"The entity is based in {entity_risk.get('country', 'Unknown')} and has been "
                        f"involved in {len(involved_patterns)} suspicious patterns.")
            
            if involved_patterns:
                narrative += " The suspicious activities include:\n"
                for pattern in involved_patterns:
                    narrative += f"- {pattern['pattern_type'].replace('_', ' ').title()} "
                    narrative += f"(Risk Score: {pattern['risk_score']:.2f})\n"
            
            sar = {
                'report_id': f"SAR-{case_id[5:]}",
                'case_id': case_id,
                'entity_id': entity_id,
                'entity_name': entity_name,
                'risk_score': case['risk_score'],
                'filing_timestamp': datetime.datetime.now().isoformat(),
                'narrative': narrative,
                'involved_patterns': involved_patterns
            }
            
        else:  # Suspicious Pattern
            pattern_type = case['pattern_type']
            pattern_details = case['pattern_details']
            
            narrative = (f"A suspicious {pattern_type.replace('_', ' ')} pattern has been detected "
                        f"with a risk score of {case['risk_score']:.2f}. ")
            
            if pattern_type == 'cycle':
                narrative += (f"The cycle starts and ends with account {pattern_details.get('start_name', 'Unknown')} "
                             f"and involves {pattern_details.get('cycle_length', 0)} transactions "
                             f"across {pattern_details.get('country_count', 0)} countries with a "
                             f"total volume of {pattern_details.get('total_volume', 0):.2f}.")
                
            elif pattern_type == 'structuring':
                narrative += (f"Account {pattern_details.get('sender_name', 'Unknown')} sent "
                             f"{pattern_details.get('transaction_count', 0)} small transactions "
                             f"to {pattern_details.get('recipient_name', 'Unknown')} over "
                             f"{pattern_details.get('days_span', 0)} days, totaling "
                             f"{pattern_details.get('total_amount', 0):.2f}.")
                
            elif pattern_type == 'layering':
                narrative += (f"The transactions were layered through {pattern_details.get('path_length', 0)} accounts "
                            f"across {pattern_details.get('country_count', 0)} countries with consistent amounts "
                            f"(ratio: {pattern_details.get('amount_consistency', 0):.2f}), suggesting deliberate "
                            f"obfuscation of fund sources.")
                
            elif pattern_type in ('fan_in', 'fan_out'):
                narrative += (f"Account {pattern_details.get('account_name', 'Unknown')} was involved in "
                            f"{pattern_details.get('connection_count', 0)} {'receiving' if pattern_type == 'fan_in' else 'sending'} "
                            f"transactions across {pattern_details.get('country_count', 0)} countries.")
                
            elif pattern_type == 'rapid_movement':
                narrative += (f"Funds moved rapidly from {pattern_details.get('start_name', 'Unknown')} to "
                            f"{pattern_details.get('end_name', 'Unknown')} through {pattern_details.get('chain_length', 0)} "
                            f"accounts in {pattern_details.get('hours_elapsed', 0):.1f} hours.")
            
            sar = {
                'report_id': f"SAR-{case_id[5:]}",
                'case_id': case_id,
                'pattern_type': pattern_type,
                'risk_score': case['risk_score'],
                'filing_timestamp': datetime.datetime.now().isoformat(),
                'narrative': narrative,
                'pattern_details': pattern_details
            }
        
        logger.info(f"Generated SAR report for case {case_id}")
        return sar
    
    def export_report(self, report_data: Dict[str, Any], format: str = 'json') -> str:
        """Export a report in the specified format"""
        if format == 'json':
            return json.dumps(report_data, indent=2)
        elif format == 'txt':
            # Generate a text version of the report
            output = []
            
            if 'report_id' in report_data:
                output.append(f"Report ID: {report_data['report_id']}")
                output.append(f"Generated: {report_data['filing_timestamp']}\n")
            
            if 'narrative' in report_data:
                output.append("NARRATIVE")
                output.append("---------")
                output.append(report_data['narrative'])
                output.append("")
            
            if 'detection_summary' in report_data:
                output.append("DETECTION SUMMARY")
                output.append("-----------------")
                summary = report_data['detection_summary']
                output.append(f"Total Patterns: {summary['total_patterns_detected']}")
                output.append("\nPattern Distribution:")
                for pattern, count in summary['pattern_distribution'].items():
                    output.append(f"- {pattern}: {count}")
                output.append("\nRisk Distribution:")
                for risk, count in summary['risk_distribution'].items():
                    output.append(f"- {risk.title()}: {count}")
                output.append("")
            
            if 'high_risk_entities' in report_data:
                output.append("HIGH RISK ENTITIES")
                output.append("-----------------")
                for entity in report_data['high_risk_entities']:
                    output.append(f"- {entity['name']} (Risk: {entity['final_risk_score']:.2f})")
                output.append("")
            
            return "\n".join(output)
        else:
            raise ValueError(f"Unsupported export format: {format}")

def main():
    """Main function to run the AML detection system"""
    # Initialize components
    data_ingestion = DataIngestion()
    kg_builder = KnowledgeGraphBuilder(use_neo4j=False)
    pattern_detector = PatternDetection(kg_builder)
    risk_scorer = RiskScoring(kg_builder, pattern_detector)
    case_manager = CaseManagement(risk_scorer, pattern_detector)
    report_generator = ReportGenerator(case_manager, risk_scorer, pattern_detector)
    
    # Run the detection pipeline
    try:
        # 1. Load and enrich data
        transactions = data_ingestion.load_from_json("sample_transactions.json")
        if not transactions:
            logger.error("No transactions loaded. Exiting.")
            return
        
        enriched_data = data_ingestion.enrich_data()
        
        # 2. Build knowledge graph
        kg_builder.build_graph(enriched_data)
        
        # 3. Detect patterns
        patterns = pattern_detector.detect_all_patterns()
        
        # 4. Calculate risk scores
        entity_risks = risk_scorer.score_all_entities()
        
        # 5. Generate cases
        cases = case_manager.generate_cases()
        
        # 6. Generate summary report
        summary_report = report_generator.generate_summary_report()
        
        # 7. Export reports
        summary_txt = report_generator.export_report(summary_report, format='txt')
        print("\nSummary Report:")
        print("==============")
        print(summary_txt)
        
        # Generate SARs for high-risk cases
        high_risk_cases = [case for case in cases if case['priority'] == 'High']
        for case in high_risk_cases[:3]:  # Process top 3 high-risk cases
            sar = report_generator.generate_suspicious_activity_report(case['case_id'])
            if sar:
                print(f"\nSuspicious Activity Report for {case['case_id']}:")
                print("==========================================")
                print(report_generator.export_report(sar, format='txt'))
        
    except Exception as e:
        logger.error(f"Error in main detection pipeline: {str(e)}")

if __name__ == "__main__":
    main()
