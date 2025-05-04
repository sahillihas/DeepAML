"""
Knowledge Graph Builder Module
----------------------------
Handles construction and querying of transaction knowledge graphs
"""

import networkx as nx
from typing import List, Dict, Any
import logging
from py2neo import Graph, Node, Relationship
import matplotlib.pyplot as plt

logger = logging.getLogger("AML_System.KnowledgeGraph")

class KnowledgeGraphBuilder:
    """Build a knowledge graph from transaction data"""
    
    def __init__(self, use_neo4j=False, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
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
        """Build a knowledge graph from transaction data"""
        if self.use_neo4j:
            return self._build_neo4j_graph(transactions)
        else:
            return self._build_networkx_graph(transactions)
    
    def _build_networkx_graph(self, transactions: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """Build a NetworkX graph from transaction data"""
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
        
        try:
            # Create constraints and indices
            self.graph.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
            self.graph.run("CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE")
            self.graph.run("CREATE INDEX account_country IF NOT EXISTS FOR (a:Account) ON (a.country)")
            self.graph.run("CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)")
        except Exception as e:
            logger.warning(f"Schema creation error: {str(e)}")
        
        # Batch process transactions
        batch_size = 100
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i+batch_size]
            
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
        
        node_count = self.graph.run("MATCH (n) RETURN count(n) AS count").data()[0]['count']
        rel_count = self.graph.run("MATCH ()-[r]->() RETURN count(r) AS count").data()[0]['count']
        
        logger.info(f"Built Neo4j graph with {node_count} nodes and {rel_count} relationships")
        return self.graph
    
    def visualize_graph(self, limit=50):
        """Visualize the graph (for NetworkX only)"""
        if self.use_neo4j:
            logger.warning("Graph visualization only available for NetworkX")
            return
        
        # Limit visualization to a subset of nodes
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
            query = """
            MATCH p = (:Account {account_id: $start})-[:SENT]->(:Transaction)-[:RECEIVED_BY]->
                    (:Account)-[:SENT*0..""" + str(max_length-1) + """]->(t:Transaction)-[:RECEIVED_BY]->
                    (:Account {account_id: $end})
            RETURN p, length(p) as path_length
            ORDER BY path_length
            LIMIT 10
            """
            return self.graph.run(query, start=start_node, end=end_node).data()
        else:
            try:
                return list(nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_length))
            except nx.NetworkXNoPath:
                return []
