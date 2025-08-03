"""
Knowledge Graph Builder
-----------------------
Constructs and queries transaction-based knowledge graphs using either NetworkX (in-memory)
or Neo4j (persistent database).
"""

import logging
from typing import List, Dict, Any, Union

import networkx as nx
from py2neo import Graph, Node, Relationship
import matplotlib.pyplot as plt

logger = logging.getLogger("AML_System.KnowledgeGraph")


class KnowledgeGraphBuilder:
    """Builds and manages a transaction knowledge graph using Neo4j or NetworkX."""

    def __init__(
        self,
        use_neo4j: bool = False,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
    ):
        self.use_neo4j = use_neo4j
        self.graph = None

        if self.use_neo4j:
            try:
                self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
                logger.info("Connected to Neo4j database.")
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}. Reverting to NetworkX.")
                self.use_neo4j = False

        if not self.use_neo4j:
            self.graph = nx.MultiDiGraph()
            logger.info("Using NetworkX for in-memory graph representation.")

    def build_graph(self, transactions: List[Dict[str, Any]]) -> Union[Graph, nx.MultiDiGraph]:
        """Builds a graph from transactions using the configured backend."""
        return self._build_neo4j_graph(transactions) if self.use_neo4j else self._build_networkx_graph(transactions)

    # --- NetworkX Graph Builder ---

    def _build_networkx_graph(self, transactions: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """Constructs a NetworkX graph from transaction data."""
        self.graph.clear()

        for tx in transactions:
            sender = tx["sender_account"]
            recipient = tx["recipient_account"]

            self.graph.add_node(sender, type="account", name=tx["sender_name"],
                                bank=tx["sender_bank"], country=tx["sender_country"])
            self.graph.add_node(recipient, type="account", name=tx["recipient_name"],
                                bank=tx["recipient_bank"], country=tx["recipient_country"])

            self.graph.add_edge(sender, recipient,
                                transaction_id=tx["transaction_id"],
                                timestamp=tx["timestamp"],
                                amount=tx["amount"],
                                currency=tx["currency"],
                                reference=tx.get("reference", ""),
                                risk_score=tx.get("risk_score", 0))

        logger.info(f"NetworkX graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    # --- Neo4j Graph Builder ---

    def _build_neo4j_graph(self, transactions: List[Dict[str, Any]]) -> Graph:
        """Constructs a Neo4j graph from transaction data."""
        try:
            self.graph.run("MATCH (n) DETACH DELETE n")
            self._setup_neo4j_schema()
        except Exception as e:
            logger.warning(f"Neo4j schema setup or reset failed: {e}")

        query = self._neo4j_transaction_query()
        batch_size = 100

        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            try:
                self.graph.run(query, transactions=batch)
            except Exception as e:
                logger.error(f"Neo4j batch insert failed (batch {i // batch_size}): {e}")

        node_count = self.graph.evaluate("MATCH (n) RETURN count(n)")
        rel_count = self.graph.evaluate("MATCH ()-[r]->() RETURN count(r)")
        logger.info(f"Neo4j graph built: {node_count} nodes, {rel_count} relationships.")
        return self.graph

    def _setup_neo4j_schema(self):
        """Creates constraints and indexes in Neo4j if not already defined."""
        self.graph.run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
        self.graph.run("CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE")
        self.graph.run("CREATE INDEX account_country IF NOT EXISTS FOR (a:Account) ON (a.country)")
        self.graph.run("CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)")

    def _neo4j_transaction_query(self) -> str:
        """Returns Cypher query string to insert transactions into Neo4j."""
        return """
        UNWIND $transactions AS tx
        MERGE (sender:Account {account_id: tx.sender_account})
          ON CREATE SET sender.name = tx.sender_name,
                        sender.bank = tx.sender_bank,
                        sender.country = tx.sender_country,
                        sender.created_at = timestamp()
        MERGE (recipient:Account {account_id: tx.recipient_account})
          ON CREATE SET recipient.name = tx.recipient_name,
                        recipient.bank = tx.recipient_bank,
                        recipient.country = tx.recipient_country,
                        recipient.created_at = timestamp()
        CREATE (t:Transaction {transaction_id: tx.transaction_id})
          SET t.timestamp = datetime(tx.timestamp),
              t.amount = tx.amount,
              t.currency = tx.currency,
              t.reference = tx.reference,
              t.risk_score = tx.risk_score,
              t.created_at = timestamp()
        CREATE (sender)-[:SENT]->(t)
        CREATE (t)-[:RECEIVED_BY]->(recipient)
        """

    # --- Visualization ---

    def visualize_graph(self, limit: int = 50):
        """Visualizes the in-memory NetworkX graph (up to `limit` nodes)."""
        if self.use_neo4j:
            logger.warning("Visualization is only available for NetworkX graphs.")
            return

        if self.graph.number_of_nodes() == 0:
            logger.warning("Cannot visualize an empty graph.")
            return

        subgraph = self.graph.subgraph(list(self.graph.nodes)[:limit])
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)

        nx.draw_networkx_nodes(subgraph, pos, node_size=700)
        nx.draw_networkx_edges(subgraph, pos, arrows=True, width=2)
        nx.draw_networkx_labels(subgraph, pos, font_size=10)

        plt.title("Transaction Knowledge Graph (Subset)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # --- Graph Querying ---

    def find_paths(self, start_node: str, end_node: str, max_length: int = 5) -> List[Any]:
        """
        Finds paths between two nodes up to a specified maximum length.
        Returns list of paths (Neo4j path objects or NetworkX path lists).
        """
        if self.use_neo4j:
            query = f"""
            MATCH p = (:Account {{account_id: $start}})-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(:Account)
                      -[:SENT*0..{max_length - 1}]->(:Transaction)-[:RECEIVED_BY]->(:Account {{account_id: $end}})
            RETURN p, length(p) AS path_length
            ORDER BY path_length
            LIMIT 10
            """
            try:
                return self.graph.run(query, start=start_node, end=end_node).data()
            except Exception as e:
                logger.error(f"Neo4j path query failed: {e}")
                return []
        else:
            try:
                return list(nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_length))
            except nx.NetworkXNoPath:
                logger.info(f"No path found from {start_node} to {end_node} within length {max_length}.")
                return []
            except nx.NodeNotFound as e:
                logger.warning(f"Node not found in graph: {e}")
                return []
