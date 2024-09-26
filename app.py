from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans_scratch import KMeans  # Importing the KMeans class from our custom implementation

app = Flask(__name__)

# Route to serve the main webpage
@app.route('/')
def index():
    return render_template('index.html')  # This renders the HTML file which we will create

# API endpoint to run the KMeans algorithm
@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    # Get the request data in JSON format
    data = request.json
    points = np.array(data['points'])  # Points for clustering
    n_clusters = int(data['n_clusters'])  # Number of clusters (K)
    init_method = data['init_method']  # Initialization method
    
    # Handle manual initialization if provided
    if init_method == 'manual':
        manual_centroids = np.array(data['manual_centroids'])
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, manual_centroids=manual_centroids)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=init_method)
    
    # Run the KMeans algorithm
    centroids, labels = kmeans.fit(points)
    
    # Return the centroids and labels as JSON
    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

