"""
Configuration settings for the DeepAML system
"""
from typing import Dict, List

# Risk Thresholds
RISK_THRESHOLDS = {
    'LOW': 0.3,
    'MEDIUM': 0.7,
    'HIGH': 0.9
}

# Transaction Pattern Detection Parameters
PATTERN_PARAMS = {
    'STRUCTURING': {
        'time_window_hours': 48,
        'min_transactions': 3,
        'max_amount_per_tx': 10000,
        'min_total_amount': 30000
    },
    'LAYERING': {
        'min_hops': 3,
        'max_time_between_hops_hours': 72,
        'min_amount_preserved': 0.8  # 80% of original amount should be preserved through the chain
    },
    'ROUND_TRIPPING': {
        'max_cycle_length': 5,
        'max_time_to_complete_hours': 168,  # One week
        'min_amount_returned': 0.9  # 90% of original amount should return
    },
    'SMURFING': {
        'min_participants': 3,
        'time_window_hours': 72,
        'min_total_amount': 50000
    },
    'MIRROR_TRADING': {
        'max_time_difference_minutes': 60,
        'amount_tolerance': 0.05,  # 5% difference in amounts
        'min_transaction_pairs': 3
    }
}

# Country Risk Classifications
COUNTRY_RISK = {
    'HIGH_RISK': ['CY', 'AE', 'MT', 'LU', 'BZ', 'PA', 'VG'],
    'MEDIUM_RISK': ['CH', 'HK', 'SG', 'MO', 'BS', 'AI'],
    'LOW_RISK': ['US', 'GB', 'DE', 'FR', 'CA', 'AU', 'JP', 'NL']
}

# Feature Extraction Settings
FEATURE_EXTRACTION = {
    'time_window_days': 30,
    'amount_percentiles': [25, 50, 75, 90, 95, 99],
    'min_transactions_for_pattern': 3,
    'graph_features': {
        'community_detection_algorithm': 'louvain',
        'centrality_measures': ['degree', 'betweenness', 'eigenvector'],
        'embedding_dimension': 128
    }
}

# Alert Generation Settings
ALERT_SETTINGS = {
    'risk_threshold_for_alert': 0.7,
    'max_alerts_per_entity_per_day': 5,
    'alert_cooldown_hours': 24,
    'batch_size': 1000
}

# Visualization Settings
VIZ_SETTINGS = {
    'dark_mode': {
        'background': '#1f2630',
        'text': '#ffffff',
        'high_risk': '#ff4444',
        'medium_risk': '#ffbb33',
        'low_risk': '#00C851',
        'edge': '#2196F3',
        'node': '#4CAF50'
    },
    'light_mode': {
        'background': '#ffffff',
        'text': '#000000',
        'high_risk': '#dc3545',
        'medium_risk': '#ffc107',
        'low_risk': '#28a745',
        'edge': '#007bff',
        'node': '#17a2b8'
    },
    'network_layout': {
        'node_size_range': [5, 20],
        'edge_width_range': [1, 5],
        'use_arrows': True,
        'show_labels': True
    }
}

# Model Parameters
MODEL_PARAMS = {
    'gnn': {
        'hidden_channels': 64,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32
    },
    'temporal': {
        'lstm_hidden_size': 128,
        'num_lstm_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    },
    'risk_scoring': {
        'feature_weights': {
            'amount': 0.3,
            'frequency': 0.2,
            'pattern_match': 0.3,
            'network_metrics': 0.2
        }
    }
}

# Dashboard Settings
DASHBOARD_CONFIG = {
    'update_interval_seconds': 300,  # 5 minutes
    'max_nodes_in_graph': 1000,
    'time_window_days': 30,
    'port': 8050,
    'debug': False
}