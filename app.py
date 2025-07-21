from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data
import requests
import numpy as np
from itertools import product
from retry import retry
import logging
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Allow all origins for development

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# GAT Model
class GATModel(torch.nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64, num_heads=4, dropout=0.1):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Load model
try:
    model = GATModel().to(device)
    model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# k-mer parameters
k = 5
bases = ['A', 'C', 'G', 'T']
all_kmers = [''.join(p) for p in product(bases, repeat=k)]
kmer_dict = {kmer: idx for idx, kmer in enumerate(all_kmers)}

# SCN2A genomic range (GRCh38)
SCN2A_START = 165239676
SCN2A_END = 165392169
CHROM2_LENGTH = 243199373  # Length of chromosome 2 (GRCh38)

@retry(tries=3, delay=1, backoff=2)
def get_sequence(chrom, start, end, strand=1):
    chrom = str(chrom).replace('chr', '')
    server = "https://rest.ensembl.org"
    ext = f"/sequence/region/human/{chrom}:{start}..{end}:{strand}?coord_system_version=GRCh38"
    headers = {"Content-Type": "text/plain"}
    r = requests.get(server + ext, headers=headers)
    if not r.ok:
        raise Exception(f"API error: {r.status_code}")
    return r.text.strip().upper()

def validate_position(chrom, pos):
    if chrom != '2':
        return False, f"Chromosome must be 2 for SCN2A, got {chrom}"
    if pos < 1 or pos > CHROM2_LENGTH:
        return False, f"Position {pos} is out of range for chromosome 2 (1-{CHROM2_LENGTH})"
    if not (SCN2A_START <= pos <= SCN2A_END):
        return False, f"Position {pos} is outside SCN2A range ({SCN2A_START}-{SCN2A_END})"
    return True, ""

def validate_sequence(seq, expected_length=201):
    if not seq or len(seq) != expected_length:
        return False
    return all(c in 'ACGT' for c in seq.upper())

def get_kmers(sequence, k=5):
    sequence = sequence.upper()
    return [sequence[i:i+k] for i in range(len(sequence)-k+1) if len(sequence[i:i+k]) == k]

def one_hot_kmer(kmer, kmer_dict):
    vector = np.zeros(len(kmer_dict))
    if kmer in kmer_dict:
        vector[kmer_dict[kmer]] = 1
    return vector

def sequence_to_graph(sequence, kmer_dict, k=5):
    kmers = get_kmers(sequence, k)
    if not kmers:
        return None
    
    node_features = []
    edge_index = []
    kmer_to_idx = {}
    node_idx = 0

    for kmer in kmers:
        if kmer not in kmer_to_idx:
            kmer_to_idx[kmer] = node_idx
            node_features.append(one_hot_kmer(kmer, kmer_dict))
            node_idx += 1

    for i in range(len(kmers)-1):
        src = kmer_to_idx[kmers[i]]
        dst = kmer_to_idx[kmers[i+1]]
        edge_index.append([src, dst])
        edge_index.append([dst, src])

    if not node_features or not edge_index:
        return None

    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index, batch=torch.zeros(node_features.shape[0], dtype=torch.long))

def predict(graph, model, device):
    graph = graph.to(device)
    with torch.no_grad():
        out = model(graph)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1).item()
        prob = probs[0][pred].item()
    return 'Pathogenic' if pred == 1 else 'Benign', prob

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        inputs = request.get_json()
        if not isinstance(inputs, list):
            return jsonify({'error': 'Input must be a list of inputs'}), 400

        results = []
        for data in inputs:
            input_type = data.get('input_type')
            label = data.get('label')
            
            if input_type == 'variant':
                chrom = data.get('chrom')
                pos = int(data.get('pos'))
                ref = data.get('ref')
                alt = data.get('alt')
                
                is_valid, error_msg = validate_position(chrom, pos)
                if not is_valid:
                    return jsonify({'error': error_msg}), 400
                
                context_window = 100
                start = max(pos - context_window, 1)
                end = pos + context_window
                seq = get_sequence(chrom, start, end)
                if not validate_sequence(seq):
                    return jsonify({'error': f'Invalid sequence fetched for chr{chrom}:{pos}'}), 400
            else:
                seq = data.get('sequence').strip().upper()
                if not validate_sequence(seq):
                    return jsonify({'error': 'Invalid sequence provided'}), 400

            graph = sequence_to_graph(seq, kmer_dict, k)
            if graph is None:
                return jsonify({'error': 'Failed to create graph'}), 400

            prediction, confidence = predict(graph, model, device)
            results.append({
                'input_type': input_type,
                'chrom': chrom if input_type == 'variant' else None,
                'pos': pos if input_type == 'variant' else None,
                'ref': ref if input_type == 'variant' else None,
                'alt': alt if input_type == 'variant' else None,
                'sequence': seq[:50] + '...' if input_type == 'sequence' else None,
                'label': label,
                'prediction': prediction,
                'confidence': confidence
            })

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
