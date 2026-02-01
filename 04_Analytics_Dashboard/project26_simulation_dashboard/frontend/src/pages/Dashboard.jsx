import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Tabs,
  Tab,
  CircularProgress,
  Alert
} from '@mui/material';
import PollutionChart from './visualizations/PollutionChart';
import TrafficChart from './visualizations/TrafficChart';
import WeatherChart from './visualizations/WeatherChart';
import AgricultureChart from './visualizations/AgricultureChart';
import SimulationList from './components/SimulationList';
import DataStream from './services/dataStream';

function Dashboard() {
  const [selectedSimulation, setSelectedSimulation] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [simulations, setSimulations] = useState([]);
  const [latestData, setLatestData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const dataStream = new DataStream();

  useEffect(() => {
    // Fetch available simulations
    fetchSimulations();

    const interval = setInterval(fetchSimulations, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedSimulation) {
      // Subscribe to real-time data
      dataStream.subscribe(selectedSimulation.id, (data) => {
        setLatestData(data);
      });

      return () => {
        dataStream.unsubscribe(selectedSimulation.id);
      };
    }
  }, [selectedSimulation]);

  const fetchSimulations = async () => {
    try {
      const response = await fetch('/api/simulations');
      const result = await response.json();
      setSimulations(result.simulations || []);
      
      // Auto-select first simulation if available
      if (result.simulations && result.simulations.length > 0 && !selectedSimulation) {
        setSelectedSimulation(result.simulations[0]);
      }
      
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch simulations: ' + err.message);
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom sx={{ mb: 4, fontWeight: 'bold' }}>
        Simulation Dashboard
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={3}>
        {/* Sidebar - Simulation List */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <SimulationList
              simulations={simulations}
              selectedSimulation={selectedSimulation}
              onSelect={setSelectedSimulation}
            />
          </Paper>
        </Grid>

        {/* Main Content - Charts */}
        <Grid item xs={12} md={9}>
          {selectedSimulation ? (
            <Box>
              <Paper sx={{ p: 2, mb: 2, bgcolor: '#f5f5f5' }}>
                <Typography variant="h5">
                  {selectedSimulation.name}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Type: {selectedSimulation.type} | Cycle: {selectedSimulation.current_cycle}
                </Typography>
              </Paper>

              <Paper>
                <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tab label="Pollution" />
                  <Tab label="Traffic" />
                  <Tab label="Weather" />
                  <Tab label="Agriculture" />
                </Tabs>

                <Box sx={{ p: 3 }}>
                  {tabValue === 0 && latestData?.pollution && (
                    <PollutionChart data={latestData} />
                  )}
                  {tabValue === 1 && latestData?.traffic && (
                    <TrafficChart data={latestData} />
                  )}
                  {tabValue === 2 && latestData?.weather && (
                    <WeatherChart data={latestData} />
                  )}
                  {tabValue === 3 && latestData?.agriculture && (
                    <AgricultureChart data={latestData} />
                  )}
                </Box>
              </Paper>
            </Box>
          ) : (
            <Alert severity="info">Select a simulation to view data</Alert>
          )}
        </Grid>
      </Grid>
    </Container>
  );
}

export default Dashboard;