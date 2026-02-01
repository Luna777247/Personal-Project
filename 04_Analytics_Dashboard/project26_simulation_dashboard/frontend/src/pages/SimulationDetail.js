import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  CircularProgress
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import PollutionChart from '../visualizations/PollutionChart';
import TrafficChart from '../visualizations/TrafficChart';
import WeatherChart from '../visualizations/WeatherChart';
import AgricultureChart from '../visualizations/AgricultureChart';
import { simulationAPI } from '../services/api';

const SimulationDetail = () => {
  const { id } = useParams();
  const [simulation, setSimulation] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const fetchSimulationData = async () => {
      try {
        const simData = await simulationAPI.getSimulation(id);
        const metricsData = await simulationAPI.getSimulationMetrics(id);

        setSimulation(simData);
        setMetrics(metricsData.metrics || {});
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch simulation data:', error);
        setLoading(false);
      }
    };

    fetchSimulationData();

    // Connect to WebSocket for real-time updates
    simulationAPI.connectToSimulation(id);
    setConnected(true);

    // Listen for updates
    const handleUpdate = (data) => {
      // Update simulation data in real-time
      setSimulation(prev => prev ? {
        ...prev,
        current_cycle: data.cycle
      } : null);
    };

    simulationAPI.onUpdate(handleUpdate);

    // Cleanup on unmount
    return () => {
      simulationAPI.disconnect();
      setConnected(false);
    };
  }, [id]);

  const handleControlSimulation = async (command) => {
    try {
      await simulationAPI.sendCommand(id, { command });
      // Refresh simulation data
      const simData = await simulationAPI.getSimulation(id);
      setSimulation(simData);
    } catch (error) {
      console.error('Failed to control simulation:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'completed': return 'info';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!simulation) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Typography variant="h6">Simulation not found</Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Simulation Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" component="h1">
            {simulation.name}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={simulation.status}
              color={getStatusColor(simulation.status)}
              size="large"
            />
            <Typography variant="body2" color="text.secondary">
              {connected ? '🟢 Connected' : '🔴 Disconnected'}
            </Typography>
          </Box>
        </Box>

        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          Type: {simulation.type} | Current Cycle: {simulation.current_cycle}
          {simulation.total_cycles && ` / ${simulation.total_cycles}`}
        </Typography>

        {/* Control Buttons */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button
            variant="contained"
            startIcon={<PlayIcon />}
            onClick={() => handleControlSimulation('start')}
            disabled={simulation.status === 'running'}
            color="success"
          >
            Start
          </Button>
          <Button
            variant="contained"
            startIcon={<PauseIcon />}
            onClick={() => handleControlSimulation('pause')}
            disabled={simulation.status !== 'running'}
            color="warning"
          >
            Pause
          </Button>
          <Button
            variant="contained"
            startIcon={<StopIcon />}
            onClick={() => handleControlSimulation('stop')}
            disabled={simulation.status === 'completed'}
            color="error"
          >
            Stop
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => handleControlSimulation('reset')}
          >
            Reset
          </Button>
        </Box>
      </Box>

      {/* Metrics Overview */}
      {Object.keys(metrics).length > 0 && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {Object.entries(metrics).map(([key, metric]) => (
            <Grid item xs={12} sm={6} md={3} key={key}>
              <Card>
                <CardContent>
                  <Typography variant="h6" component="h3" sx={{ mb: 1 }}>
                    {key.replace('_', ' ').toUpperCase()}
                  </Typography>
                  <Typography variant="h4" color="primary" sx={{ mb: 1 }}>
                    {typeof metric.current_value === 'number' ? metric.current_value.toFixed(2) : metric.current_value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Min: {metric.min_value?.toFixed(2)} | Max: {metric.max_value?.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Visualization Charts */}
      <Typography variant="h5" component="h2" sx={{ mb: 3 }}>
        Real-time Visualizations
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Pollution Monitoring
              </Typography>
              <PollutionChart simulationId={id} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Traffic Analysis
              </Typography>
              <TrafficChart simulationId={id} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Weather Data
              </Typography>
              <WeatherChart simulationId={id} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Agricultural Metrics
              </Typography>
              <AgricultureChart simulationId={id} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SimulationDetail;