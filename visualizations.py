"""
Visualization Module
-----------------
Handles visualization of transaction data and patterns
"""

import os
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
from typing import List, Dict, Any, Optional, Tuple
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.subplots as sp
import datetime
import signal
import sys
import logging
from flask import request
from models import PatternType, RiskLevel

logger = logging.getLogger("AML_System.Visualizations")

class TransactionVisualizer:
    def __init__(self, dark_mode: bool = True):
        """Initialize visualizer with color scheme"""
        self.dark_mode = dark_mode
        self.color_scheme = {
            'background': '#1f2630' if dark_mode else '#ffffff',
            'text': '#ffffff' if dark_mode else '#000000',
            'high_risk': '#ff4444',
            'medium_risk': '#ffbb33',
            'low_risk': '#00C851',
            'edge': '#2196F3',
            'node': '#4CAF50',
            'pattern_colors': {
                PatternType.STRUCTURING.value: '#FF6B6B',
                PatternType.LAYERING.value: '#4ECDC4',
                PatternType.ROUND_TRIP.value: '#45B7D1',
                PatternType.RAPID_MOVEMENT.value: '#96CEB4',
                PatternType.SMURFING.value: '#FFEEAD',
                PatternType.FAN_IN.value: '#D4A5A5',
                PatternType.FAN_OUT.value: '#9AC1D9',
                PatternType.UNKNOWN.value: '#858585'
            }
        }
        self.app = None
        self._setup_signal_handlers()
        logger.info("Visualization module initialized")

    def _setup_signal_handlers(self):
    """Set up signal handlers for graceful shutdown of the visualization server."""

    def signal_handler(sig, frame):
        logger.info("Received signal %s. Initiating graceful shutdown...", sig)

        try:
            if self.app:
                shutdown_func = None
                try:
                    from flask import request
                    shutdown_func = request.environ.get('werkzeug.server.shutdown')
                except RuntimeError:
                    logger.warning("No active request context. Using sys.exit instead.")

                if shutdown_func:
                    shutdown_func()
                else:
                    logger.info("No shutdown function found. Exiting process.")
            else:
                logger.info("No application instance found. Exiting process.")
        except Exception as e:
            logger.error("Error during shutdown: %s", e)
        finally:
            sys.exit(0)


    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        if risk_score >= 0.7:
            return self.color_scheme['high_risk']
        elif risk_score >= 0.5:
            return self.color_scheme['medium_risk']
        return self.color_scheme['low_risk']

    def _get_pattern_color(self, pattern_type: str) -> str:
        """Get color for pattern type"""
        return self.color_scheme['pattern_colors'].get(pattern_type, self.color_scheme['pattern_colors']['unknown'])

    def create_transaction_network(self, transactions: List[Dict], output_path: str):
        """Create an interactive network visualization of transactions"""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create network
        net = Network(height='750px', width='100%', bgcolor=self.color_scheme['background'],
                     font_color=self.color_scheme['text'])
        net.force_atlas_2based()

        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for tx in transactions:
            sender_id = tx['sender']
            receiver_id = tx['receiver']
            
            # Add nodes if they don't exist
            if not G.has_node(sender_id):
                G.add_node(sender_id, title=f"Account: {tx['sender_name']}\nCountry: {tx['sender_country']}")
            if not G.has_node(receiver_id):
                G.add_node(receiver_id, title=f"Account: {tx['receiver_name']}\nCountry: {tx['receiver_country']}")
            
            # Add edge
            G.add_edge(sender_id, receiver_id, 
                      title=f"Amount: {tx['amount']:,.2f} {tx['currency']}\nDate: {tx['timestamp']}")

        # Add nodes to visualization
        for node_id in G.nodes():
            net.add_node(node_id, label=node_id, title=G.nodes[node_id]['title'])

        # Add edges to visualization
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], title=edge[2]['title'])

        # Save visualization
        net.save_graph(output_path)
        return G

    def create_pattern_visualization(self, pattern_data: Dict[str, List[Dict]], pattern_type: str):
        """Create a specialized visualization for a specific pattern type"""
        if pattern_type == 'structuring':
            return self._visualize_structuring(pattern_data['transactions'])
        elif pattern_type == 'layering':
            return self._visualize_layering(pattern_data['transactions'])
        elif pattern_type == 'round_tripping':
            return self._visualize_round_tripping(pattern_data['transactions'])
        elif pattern_type == 'smurfing':
            return self._visualize_smurfing(pattern_data['transactions'])
        else:
            return self._visualize_generic_pattern(pattern_data['transactions'])

    def _visualize_structuring(self, transactions: List[Dict]):
        """Visualize structuring pattern"""
        # Convert transactions to DataFrame
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure with secondary y-axis
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add scatter plot for individual transactions
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['amount'],
                      mode='markers',
                      name='Transactions',
                      marker=dict(size=10,
                                color=self.color_scheme['edge'])),
            secondary_y=False,
        )
        
        # Add cumulative sum line
        df = df.sort_values('timestamp')
        df['cumulative_sum'] = df['amount'].cumsum()
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cumulative_sum'],
                      mode='lines+markers',
                      name='Cumulative Amount',
                      line=dict(color=self.color_scheme['high_risk'])),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title='Structuring Pattern Analysis',
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig

    def _visualize_layering(self, transactions: List[Dict]):
        """Visualize layering pattern"""
        # Create Sankey diagram
        df = pd.DataFrame(transactions)
        
        # Create nodes
        all_entities = pd.concat([
            df[['sender', 'sender_name']].rename(columns={'sender': 'id', 'sender_name': 'name'}),
            df[['receiver', 'receiver_name']].rename(columns={'receiver': 'id', 'receiver_name': 'name'})
        ]).drop_duplicates()
        
        node_ids = {node: idx for idx, node in enumerate(all_entities['id'])}
        
        # Create links
        links = {
            'source': [node_ids[sender] for sender in df['sender']],
            'target': [node_ids[receiver] for receiver in df['receiver']],
            'value': df['amount'],
            'label': df['amount'].apply(lambda x: f'{x:,.2f}')
        }
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_entities['name'],
                color=self.color_scheme['node']
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                label=links['label'],
                color=self.color_scheme['edge']
            )
        )])
        
        fig.update_layout(
            title='Transaction Layering Flow',
            font=dict(color=self.color_scheme['text']),
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background']
        )
        
        return fig

    def _visualize_round_tripping(self, transactions: List[Dict]):
        """Visualize round-tripping pattern"""
        # Create circular network layout
        G = nx.DiGraph()
        
        # Add nodes and edges
        for tx in transactions:
            G.add_edge(tx['sender'], tx['receiver'], 
                      amount=tx['amount'],
                      timestamp=tx['timestamp'])
        
        # Create positions using circular layout
        pos = nx.circular_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Amount: {edge[2]['amount']:,.2f}\n"
                           f"Time: {edge[2]['timestamp']}")
        
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=self.color_scheme['edge']),
            hoverinfo='text',
            text=edge_text,
            mode='lines+markers',
            name='Transactions'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Account: {node}")
        
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=20,
                color=self.color_scheme['node']
            ),
            name='Accounts'
        )
        
        # Create figure
        fig = go.Figure(data=[edges_trace, nodes_trace])
        
        fig.update_layout(
            title='Round-Tripping Pattern',
            showlegend=True,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text']),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig

    def _visualize_smurfing(self, transactions: List[Dict]):
        """Visualize smurfing pattern"""
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for transactions
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['amount'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['amount'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df.apply(lambda row: f"From: {row['sender_name']}<br>"
                                        f"Amount: {row['amount']:,.2f} {row['currency']}<br>"
                                        f"To: {row['receiver_name']}", axis=1),
                hoverinfo='text'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Smurfing Pattern Analysis',
            xaxis_title='Time',
            yaxis_title='Transaction Amount',
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig

    def _visualize_generic_pattern(self, transactions: List[Dict]):
        """Create a generic visualization for other patterns"""
        df = pd.DataFrame(transactions)
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for _, tx in df.iterrows():
            G.add_edge(tx['sender'], tx['receiver'], 
                      amount=tx['amount'],
                      timestamp=tx['timestamp'])
        
        # Create positions using spring layout
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Amount: {edge[2]['amount']:,.2f}\n"
                           f"Time: {edge[2]['timestamp']}")
        
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=self.color_scheme['edge']),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Account: {node}")
        
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=self.color_scheme['node']
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edges_trace, nodes_trace])
        
        fig.update_layout(
            title='Transaction Pattern Analysis',
            showlegend=False,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig

    def create_dashboard(self, transactions: List[Dict], pattern_results: Dict[str, List[Dict]], port: int = 8050):
        """Create an interactive dashboard for transaction analysis"""
        os.makedirs('visualizations', exist_ok=True)
        
        self.app = dash.Dash(__name__, 
                           external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        
        self.app.layout = html.Div(style={'backgroundColor': self.color_scheme['background']}, children=[
            html.H1('AML Transaction Analysis Dashboard',
                   style={'color': self.color_scheme['text'], 'textAlign': 'center'}),
            
            # Time range selector
            html.Div([
                html.H3('Select Time Range',
                       style={'color': self.color_scheme['text']}),
                dcc.RangeSlider(
                    id='time-slider',
                    min=df['timestamp'].min().timestamp(),
                    max=df['timestamp'].max().timestamp(),
                    value=[df['timestamp'].min().timestamp(), df['timestamp'].max().timestamp()],
                    marks={int(ts.timestamp()): ts.strftime('%Y-%m-%d')
                          for ts in pd.date_range(start=df['timestamp'].min(),
                                                end=df['timestamp'].max(),
                                                freq='W')},
                    step=86400  # One day in seconds
                )
            ], style={'margin': '20px'}),
            
            # Overview Section
            html.Div([
                html.H2('Overview',
                       style={'color': self.color_scheme['text']}),
                html.Div([
                    dcc.Graph(id='risk-timeline'),
                    dcc.Graph(id='pattern-summary')
                ], style={'display': 'flex'})
            ]),
            
            # Network Analysis Section
            html.Div([
                html.H2('Network Analysis',
                       style={'color': self.color_scheme['text']}),
                dcc.Tabs([
                    dcc.Tab(label='2D Network',
                           children=[dcc.Graph(id='transaction-network')]),
                    dcc.Tab(label='3D Network',
                           children=[dcc.Graph(id='network-3d')]),
                    dcc.Tab(label='Country Risk',
                           children=[dcc.Graph(id='country-risk')])
                ])
            ]),
            
            # Pattern Analysis Section
            html.Div([
                html.H2('Pattern Analysis',
                       style={'color': self.color_scheme['text']}),
                dcc.Tabs([
                    dcc.Tab(label='Structuring',
                           children=[dcc.Graph(id='structuring-pattern')]),
                    dcc.Tab(label='Layering',
                           children=[dcc.Graph(id='layering-pattern')]),
                    dcc.Tab(label='Round-Trip',
                           children=[dcc.Graph(id='round-trip-pattern')]),
                    dcc.Tab(label='Smurfing',
                           children=[dcc.Graph(id='smurfing-pattern')])
                ])
            ])
        ])

        @self.app.callback(
            [Output('risk-timeline', 'figure'),
             Output('pattern-summary', 'figure'),
             Output('transaction-network', 'figure'),
             Output('network-3d', 'figure'),
             Output('country-risk', 'figure'),
             Output('structuring-pattern', 'figure'),
             Output('layering-pattern', 'figure'),
             Output('round-trip-pattern', 'figure'),
             Output('smurfing-pattern', 'figure')],
            [Input('time-slider', 'value')]
        )
        def update_figures(time_range):
            # Filter data based on time range
            mask = (df['timestamp'] >= datetime.datetime.fromtimestamp(time_range[0])) & \
                  (df['timestamp'] <= datetime.datetime.fromtimestamp(time_range[1]))
            filtered_df = df[mask]
            
            # Get pattern visualizations
            pattern_figs = self._update_dashboard_pattern_view(filtered_df, pattern_results)
            
            # Transaction networks (2D and 3D)
            network_2d = self._create_network_figure(filtered_df)
            network_3d = self.create_3d_network(filtered_df.to_dict('records'), pattern_results)
            
            # Country risk distribution with risk coloring
            country_risks = filtered_df.groupby('sender_country').agg({
                'amount': 'sum',
                'risk_score': 'mean'
            }).reset_index()
            
            country_fig = go.Figure(data=[
                go.Bar(
                    x=country_risks['sender_country'],
                    y=country_risks['amount'],
                    marker_color=[self._get_risk_color(score) for score in country_risks['risk_score']],
                    text=country_risks['risk_score'].apply(lambda x: f'Risk: {x:.2f}'),
                    textposition='auto',
                )
            ])
            
            country_fig.update_layout(
                title='Transaction Volume and Risk by Country',
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor=self.color_scheme['background'],
                font=dict(color=self.color_scheme['text'])
            )
            
            return (
                pattern_figs['risk_timeline'],
                pattern_figs['pattern_summary'],
                network_2d,
                network_3d,
                country_fig,
                pattern_figs.get(f'pattern_{PatternType.STRUCTURING.value}', self._create_empty_figure('No structuring patterns detected')),
                pattern_figs.get(f'pattern_{PatternType.LAYERING.value}', self._create_empty_figure('No layering patterns detected')),
                pattern_figs.get(f'pattern_{PatternType.ROUND_TRIP.value}', self._create_empty_figure('No round-trip patterns detected')),
                pattern_figs.get(f'pattern_{PatternType.SMURFING.value}', self._create_empty_figure('No smurfing patterns detected'))
            )

        try:
            print("\nDashboard is running at: http://127.0.0.1:8050")
            print("Press Ctrl+C to stop the server")
            self.app.run(debug=False, port=port, host='127.0.0.1', use_reloader=False)
        except Exception as e:
            print(f"Error running dashboard: {str(e)}")
            
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.color_scheme['text'])
        )
        fig.update_layout(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background']
        )
        return fig

    def _create_network_figure(self, df: pd.DataFrame) -> go.Figure:
        """Helper function to create network visualization for the dashboard"""
        G = nx.DiGraph()
        
        # Add nodes and edges with risk information
        for _, row in df.iterrows():
            sender_id = row['sender']
            receiver_id = row['receiver']
            
            # Add nodes if they don't exist
            if not G.has_node(sender_id):
                G.add_node(sender_id, 
                          name=row['sender_name'], 
                          country=row['sender_country'],
                          risk_score=row.get('risk_score', 0))
            if not G.has_node(receiver_id):
                G.add_node(receiver_id, 
                          name=row['receiver_name'], 
                          country=row['receiver_country'],
                          risk_score=row.get('risk_score', 0))
            
            # Add edge with risk information
            G.add_edge(sender_id, receiver_id, 
                      amount=row['amount'],
                      risk_score=row.get('risk_score', 0))
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G)
        
        # Create edges trace with risk-based coloring
        edge_x = []
        edge_y = []
        edge_text = []
        
        edges_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            risk_score = edge[2].get('risk_score', 0)
            
            edges_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color=self._get_risk_color(risk_score)),
                hoverinfo='text',
                text=f"Amount: {edge[2]['amount']:,.2f}<br>Risk Score: {risk_score:.2f}",
                mode='lines',
                showlegend=False
            ))
        
        # Create nodes trace with risk-based sizing
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node, attr in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            risk_score = attr.get('risk_score', 0.1)  # Default to low risk if missing
            node_text.append(
                f"Account: {attr['name']}<br>"
                f"Country: {attr['country']}<br>"
                f"Risk Score: {risk_score:.2f}"
            )
            node_colors.append(self._get_risk_color(risk_score))
            node_sizes.append(max(10, 10 + risk_score * 20))  # Ensure minimum size of 10
        
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2)),
            name='Accounts'
        )
        
        # Create figure with improved layout
        fig = go.Figure(data=edges_trace + [nodes_trace],
                       layout=go.Layout(
                           title=dict(
                               text='Transaction Network (Size and Color by Risk)',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           plot_bgcolor=self.color_scheme['background'],
                           paper_bgcolor=self.color_scheme['background'],
                           font=dict(color=self.color_scheme['text']),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        return fig

    def create_pattern_summary(self, patterns: Dict[str, List[Dict]]) -> go.Figure:
        """Create a summary visualization of all detected patterns"""
        pattern_counts = {}
        pattern_risks = {}
        
        for pattern_type, pattern_list in patterns.items():
            pattern_counts[pattern_type] = len(pattern_list)
            if pattern_list:
                pattern_risks[pattern_type] = np.mean([p.get('risk_score', 0) for p in pattern_list])
            else:
                pattern_risks[pattern_type] = 0
        
        fig = sp.make_subplots(rows=1, cols=2, 
                              subplot_titles=('Pattern Distribution', 'Average Risk by Pattern'))
        
        # Pattern distribution
        fig.add_trace(
            go.Bar(
                x=list(pattern_counts.keys()),
                y=list(pattern_counts.values()),
                marker_color=[self._get_pattern_color(pt) for pt in pattern_counts.keys()],
                name='Pattern Count'
            ),
            row=1, col=1
        )
        
        # Risk distribution
        fig.add_trace(
            go.Bar(
                x=list(pattern_risks.keys()),
                y=list(pattern_risks.values()),
                marker_color=[self._get_risk_color(risk) for risk in pattern_risks.values()],
                name='Risk Score'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Pattern Analysis Summary',
            showlegend=True,
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig

    def visualize_risk_timeline(self, transactions: List[Dict], pattern_results: Dict[str, List[Dict]]) -> go.Figure:
        """Create a timeline visualization of transaction risks"""
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate risk scores for each transaction
        risk_scores = []
        pattern_types = []
        
        for _, tx in df.iterrows():
            # Find matching patterns for this transaction
            tx_patterns = []
            for pattern_type, patterns in pattern_results.items():
                for pattern in patterns:
                    if (tx['sender'] in pattern.get('accounts', []) or 
                        tx['receiver'] in pattern.get('accounts', [])):
                        tx_patterns.append((pattern_type, pattern.get('risk_score', 0)))
            
            if tx_patterns:
                # Use highest risk score if multiple patterns
                pattern_type, risk_score = max(tx_patterns, key=lambda x: x[1])
            else:
                pattern_type, risk_score = 'normal', 0.1
                
            risk_scores.append(risk_score)
            pattern_types.append(pattern_type)
        
        df['risk_score'] = risk_scores
        df['pattern_type'] = pattern_types
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for transactions
        for pattern in df['pattern_type'].unique():
            mask = df['pattern_type'] == pattern
            fig.add_trace(
                go.Scatter(
                    x=df[mask]['timestamp'],
                    y=df[mask]['risk_score'],
                    mode='markers',
                    name=pattern,
                    marker=dict(
                        size=10,
                        color=[self._get_pattern_color(p) for p in df[mask]['pattern_type']]
                    ),
                    text=df[mask].apply(
                        lambda row: f"From: {row['sender']}<br>"
                                  f"To: {row['receiver']}<br>"
                                  f"Amount: {row['amount']:,.2f}<br>"
                                  f"Risk: {row['risk_score']:.2f}",
                        axis=1
                    ),
                    hoverinfo='text'
                )
            )
        
        # Add risk threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color=self.color_scheme['high_risk'],
                     annotation_text="High Risk", annotation_position="right")
        fig.add_hline(y=0.5, line_dash="dash", line_color=self.color_scheme['medium_risk'],
                     annotation_text="Medium Risk", annotation_position="right")
        
        fig.update_layout(
            title='Transaction Risk Timeline',
            xaxis_title='Time',
            yaxis_title='Risk Score',
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text']),
            showlegend=True
        )
        
        return fig

    def _update_dashboard_pattern_view(self, df: pd.DataFrame, pattern_results: Dict[str, List[Dict]]) -> Dict[str, go.Figure]:
        """Update dashboard with pattern-specific visualizations"""
        figs = {}
        
        # Pattern summary
        figs['pattern_summary'] = self.create_pattern_summary(pattern_results)
        
        # Risk timeline
        figs['risk_timeline'] = self.visualize_risk_timeline(df.to_dict('records'), pattern_results)
        
        # Pattern-specific visualizations
        for pattern_type, patterns in pattern_results.items():
            if patterns:
                if pattern_type == PatternType.STRUCTURING.value:
                    figs[f'pattern_{pattern_type}'] = self._visualize_structuring(patterns)
                elif pattern_type == PatternType.LAYERING.value:
                    figs[f'pattern_{pattern_type}'] = self._visualize_layering(patterns)
                elif pattern_type == PatternType.ROUND_TRIP.value:
                    figs[f'pattern_{pattern_type}'] = self._visualize_round_tripping(patterns)
                elif pattern_type == PatternType.SMURFING.value:
                    figs[f'pattern_{pattern_type}'] = self._visualize_smurfing(patterns)
        
        return figs

    def create_3d_network(self, transactions: List[Dict], pattern_results: Dict[str, List[Dict]]) -> go.Figure:
        """Create an interactive 3D network visualization"""
        G = nx.DiGraph()
        
        # Add nodes and edges with risk and pattern information
        for tx in transactions:
            sender_id = tx['sender']
            receiver_id = tx['receiver']
            
            # Find patterns involving these accounts
            tx_patterns = []
            for pattern_type, patterns in pattern_results.items():
                for pattern in patterns:
                    if (sender_id in pattern.get('accounts', []) or 
                        receiver_id in pattern.get('accounts', [])):
                        tx_patterns.append((pattern_type, pattern.get('risk_score', 0)))
            
            # Get highest risk pattern
            if tx_patterns:
                pattern_type, risk_score = max(tx_patterns, key=lambda x: x[1])
            else:
                pattern_type, risk_score = 'normal', 0.1
            
            # Add nodes if they don't exist
            for node_id, name, country in [(sender_id, tx['sender_name'], tx['sender_country']),
                                         (receiver_id, tx['receiver_name'], tx['receiver_country'])]:
                if not G.has_node(node_id):
                    G.add_node(node_id,
                             name=name,
                             country=country,
                             risk_score=risk_score,
                             pattern_type=pattern_type)
            
            # Add edge
            G.add_edge(sender_id, receiver_id,
                      amount=tx['amount'],
                      risk_score=risk_score,
                      pattern_type=pattern_type,
                      timestamp=tx['timestamp'])
        
        # Create 3D layout
        pos_3d = nx.spring_layout(G, dim=3)
        
        # Create edge traces
        edge_traces = []
        
        for edge in G.edges(data=True):
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            
            # Create Bezier curve control point
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            mid_z = (z0 + z1) / 2 + 0.1  # Lift the curve slightly
            
            # Create curve points
            curve_x = np.array([x0, mid_x, x1])
            curve_y = np.array([y0, mid_y, y1])
            curve_z = np.array([z0, mid_z, z1])
            
            # Get pattern color and risk color
            pattern_color = self._get_pattern_color(edge[2]['pattern_type'])
            risk_color = self._get_risk_color(edge[2]['risk_score'])
            
            # Create edge trace with gradient color
            edge_traces.append(go.Scatter3d(
                x=curve_x, y=curve_y, z=curve_z,
                mode='lines',
                line=dict(
                    color=[pattern_color, risk_color],
                    width=2
                ),
                hoverinfo='text',
                text=f"Amount: {edge[2]['amount']:,.2f}<br>"
                     f"Pattern: {edge[2]['pattern_type']}<br>"
                     f"Risk Score: {edge[2]['risk_score']:.2f}",
                showlegend=False
            ))
        
        # Create node trace
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node, attr in G.nodes(data=True):
            x, y, z = pos_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(
                f"Account: {attr['name']}<br>"
                f"Country: {attr['country']}<br>"
                f"Pattern: {attr['pattern_type']}<br>"
                f"Risk Score: {attr['risk_score']:.2f}"
            )
            node_colors.append(self._get_risk_color(attr['risk_score']))
            node_sizes.append(10 + attr['risk_score'] * 20)
        
        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1),
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text',
            name='Accounts'
        )
        
        # Create figure
        data = edge_traces + [nodes_trace]
        
        layout = go.Layout(
            title='3D Transaction Network Analysis',
            showlegend=True,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                dragmode='orbit'
            ),
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return go.Figure(data=data, layout=layout)
