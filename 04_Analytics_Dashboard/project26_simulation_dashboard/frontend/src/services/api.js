import io from 'socket.io-client';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class SimulationAPI {
  constructor() {
    this.socket = null;
    this.listeners = {};
  }

  // REST API methods
  async getSimulations() {
    const response = await fetch(`${API_BASE_URL}/api/simulations`);
    if (!response.ok) {
      throw new Error('Failed to fetch simulations');
    }
    return response.json();
  }

  async getSimulation(simulationId) {
    const response = await fetch(`${API_BASE_URL}/api/simulations/${simulationId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch simulation');
    }
    return response.json();
  }

  async getSimulationData(simulationId, limit = 100) {
    const response = await fetch(`${API_BASE_URL}/api/data/${simulationId}?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to fetch simulation data');
    }
    return response.json();
  }

  async getSimulationMetrics(simulationId) {
    const response = await fetch(`${API_BASE_URL}/api/metrics/${simulationId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch simulation metrics');
    }
    return response.json();
  }

  async createSimulation(config) {
    const response = await fetch(`${API_BASE_URL}/api/simulations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      throw new Error('Failed to create simulation');
    }
    return response.json();
  }

  async sendCommand(simulationId, command) {
    const response = await fetch(`${API_BASE_URL}/api/simulations/${simulationId}/control`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(command),
    });
    if (!response.ok) {
      throw new Error('Failed to send command');
    }
    return response.json();
  }

  // WebSocket methods
  connectToSimulation(simulationId) {
    if (this.socket) {
      this.socket.disconnect();
    }

    this.socket = io(`${API_BASE_URL}`, {
      query: { simulationId }
    });

    this.socket.on('connect', () => {
      console.log('Connected to simulation:', simulationId);
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from simulation:', simulationId);
    });

    this.socket.on('update', (data) => {
      this.notifyListeners('update', data);
    });

    this.socket.on('initial_data', (data) => {
      this.notifyListeners('initial_data', data);
    });

    this.socket.on('error', (error) => {
      this.notifyListeners('error', error);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // Event listener methods
  addListener(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  removeListener(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  notifyListeners(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }

  // Utility methods
  onUpdate(callback) {
    this.addListener('update', callback);
  }

  onInitialData(callback) {
    this.addListener('initial_data', callback);
  }

  onError(callback) {
    this.addListener('error', callback);
  }
}

// Create singleton instance
export const simulationAPI = new SimulationAPI();
export default simulationAPI;