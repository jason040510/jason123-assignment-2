from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans_scratch import KMeans

app = Flask(__name__)

# Variable to store the current state of the KMeans algorithm
kmeans_state = {}

@app.route('/')
def index():
    return render_template('index.html')

# Initialize the KMeans algorithm for step-wise execution
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    global kmeans_state
    data = request.json
    points = np.array(data['points'])
    n_clusters = int(data['n_clusters'])
    init_method = data['init_method']
    manual_centroids = np.array(data['manual_centroids']) if 'manual_centroids' in data else None
    if init_method == 'manual' and manual_centroids is not None:
        manual_centroids = np.array(data['manual_centroids'])
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, manual_centroids=manual_centroids)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=init_method)

    # Initialize KMeans for step-wise execution
    kmeans_state = {
        'kmeans': kmeans,
        'points': points,
        'iteration': 0,
        'converged': False
    }

    # Run the first step
    centroids, labels, converged = kmeans.partial_fit(points)
    kmeans_state['converged'] = converged

    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist()})

# Step through the KMeans algorithm
@app.route('/step_kmeans', methods=['POST'])
def step_kmeans():
    global kmeans_state

    if kmeans_state.get('converged', False):
        return jsonify({'message': 'Already converged!', 'converged': True})

    kmeans = kmeans_state['kmeans']
    points = kmeans_state['points']

    # Perform one step of the KMeans algorithm
    centroids, labels, converged = kmeans.partial_fit(points)

    kmeans_state['converged'] = converged  # Update convergence status
    kmeans_state['iteration'] += 1  # Track the current iteration

    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist(), 'converged': converged})

# Run the KMeans algorithm to full convergence
@app.route('/run_to_convergence', methods=['POST'])
def run_to_convergence():
    global kmeans_state

    kmeans = kmeans_state['kmeans']
    points = kmeans_state['points']

    # Run KMeans to full convergence
    centroids, labels = kmeans.fit(points)
    kmeans_state['converged'] = True  # Mark the algorithm as converged

    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist(), 'converged': True})

# Reset the KMeans state
@app.route('/reset', methods=['POST'])
def reset():
    global kmeans_state
    kmeans_state.clear()
    return jsonify({'message': 'KMeans state has been reset'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
