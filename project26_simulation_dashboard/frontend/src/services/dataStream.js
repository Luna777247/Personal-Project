/**
 * WebSocket Data Stream Service
 * Manages real-time connections to backend simulations
 */

class DataStream {
  constructor() {
    this.connections = {};
    this.subscribers = {};
  }

  /**
   * Subscribe to real-time data from a simulation
   * @param {string} simulationId - The simulation ID
   * @param {function} callback - Callback function for data updates
   */
  subscribe(simulationId, callback) {
    if (!this.subscribers[simulationId]) {
      this.subscribers[simulationId] = [];
      this.connect(simulationId);
    }
    this.subscribers[simulationId].push(callback);
  }

  /**
   * Unsubscribe from a simulation
   * @param {string} simulationId - The simulation ID
   */
  unsubscribe(simulationId) {
    delete this.subscribers[simulationId];
    if (this.connections[simulationId]) {
      this.connections[simulationId].close();
      delete this.connections[simulationId];
    }
  }

  /**
   * Connect to simulation WebSocket
   * @param {string} simulationId - The simulation ID
   */
  connect(simulationId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/simulations/${simulationId}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`Connected to simulation ${simulationId}`);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (this.subscribers[simulationId]) {
            this.subscribers[simulationId].forEach(callback => {
              callback(message.data);
            });
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (error) => {
        console.error(`WebSocket error for simulation ${simulationId}:`, error);
      };

      ws.onclose = () => {
        console.log(`Disconnected from simulation ${simulationId}`);
        // Attempt reconnect after 3 seconds
        setTimeout(() => {
          if (this.subscribers[simulationId]) {
            this.connect(simulationId);
          }
        }, 3000);
      };

      this.connections[simulationId] = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
    }
  }
}

export default DataStream;