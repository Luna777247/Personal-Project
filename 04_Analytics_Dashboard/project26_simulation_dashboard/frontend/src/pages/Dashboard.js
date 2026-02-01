import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Add as AddIcon
} from '@mui/icons-material';

import PollutionChart from '../visualizations/PollutionChart';
import TrafficChart from '../visualizations/TrafficChart';
import WeatherChart from '../visualizations/WeatherChart';
import AgricultureChart from '../visualizations/AgricultureChart';
import { simulationAPI } from '../services/api';

const Dashboard = () => {
  const [simulations, setSimulations] = useState([]);
  const [selectedSimulation, setSelectedSimulation] = useState(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newSimulation, setNewSimulation] = useState({
    name: '',
    type: 'climate',
    description: ''
  });

  useEffect(() => {
    const fetchSimulations = async () => {
      try {
        const data = await simulationAPI.getSimulations();
        setSimulations(data.simulations || []);
        // Auto-select the first running simulation
        const runningSim = data.simulations?.find(sim => sim.status === 'running');
        if (runningSim) {
          setSelectedSimulation(runningSim.id);
        } else if (data.simulations?.length > 0) {
          setSelectedSimulation(data.simulations[0].id);
        }
      } catch (error) {
        console.error('Failed to fetch simulations:', error);
      }
    };

    fetchSimulations();
  }, []);

  const handleCreateSimulation = async () => {
    try {
      await simulationAPI.createSimulation(newSimulation);
      setCreateDialogOpen(false);
      setNewSimulation({ name: '', type: 'climate', description: '' });
      // Refresh simulations list
      const data = await simulationAPI.getSimulations();
      setSimulations(data.simulations || []);
    } catch (error) {
      console.error('Failed to create simulation:', error);
    }
  };

  const handleControlSimulation = async (simulationId, command) => {
    try {
      await simulationAPI.sendCommand(simulationId, { command });
      // Refresh simulations list
      const data = await simulationAPI.getSimulations();
      setSimulations(data.simulations || []);
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

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Simulation Dashboard
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          New Simulation
        </Button>
      </Box>

      {/* Active Simulations Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {simulations.map((simulation) => (
          <Grid item xs={12} md={6} lg={4} key={simulation.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Typography variant="h6" component="h2">
                    {simulation.name}
                  </Typography>
                  <Chip
                    label={simulation.status}
                    color={getStatusColor(simulation.status)}
                    size="small"
                  />
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Type: {simulation.type} | Cycle: {simulation.current_cycle}
                </Typography>

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<PlayIcon />}
                    onClick={() => handleControlSimulation(simulation.id, 'start')}
                    disabled={simulation.status === 'running'}
                  >
                    Start
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<PauseIcon />}
                    onClick={() => handleControlSimulation(simulation.id, 'pause')}
                    disabled={simulation.status !== 'running'}
                  >
                    Pause
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<StopIcon />}
                    onClick={() => handleControlSimulation(simulation.id, 'stop')}
                    disabled={simulation.status === 'completed'}
                  >
                    Stop
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Visualization Charts */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2">
          Real-time Metrics
        </Typography>
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel>Select Simulation</InputLabel>
          <Select
            value={selectedSimulation || ''}
            label="Select Simulation"
            onChange={(e) => setSelectedSimulation(e.target.value)}
          >
            {simulations.map((simulation) => (
              <MenuItem key={simulation.id} value={simulation.id}>
                {simulation.name} ({simulation.status})
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Pollution Monitoring
              </Typography>
              <PollutionChart simulationId={selectedSimulation} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Traffic Analysis
              </Typography>
              <TrafficChart simulationId={selectedSimulation} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Weather Data
              </Typography>
              <WeatherChart simulationId={selectedSimulation} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
                Agricultural Metrics
              </Typography>
              <AgricultureChart simulationId={selectedSimulation} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Create Simulation Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Simulation</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Simulation Name"
            fullWidth
            variant="outlined"
            value={newSimulation.name}
            onChange={(e) => setNewSimulation({ ...newSimulation, name: e.target.value })}
            sx={{ mb: 2, mt: 1 }}
          />

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Simulation Type</InputLabel>
            <Select
              value={newSimulation.type}
              label="Simulation Type"
              onChange={(e) => setNewSimulation({ ...newSimulation, type: e.target.value })}
            >
              <MenuItem value="climate">Climate Change</MenuItem>
              <MenuItem value="pollution">Urban Pollution</MenuItem>
              <MenuItem value="traffic">Traffic Analysis</MenuItem>
              <MenuItem value="agriculture">Agricultural Impact</MenuItem>
            </Select>
          </FormControl>

          <TextField
            margin="dense"
            label="Description (Optional)"
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            value={newSimulation.description}
            onChange={(e) => setNewSimulation({ ...newSimulation, description: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateSimulation} variant="contained">
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Dashboard;