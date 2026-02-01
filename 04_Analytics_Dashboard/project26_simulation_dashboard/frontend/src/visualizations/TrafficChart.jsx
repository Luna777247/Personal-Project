import React from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function TrafficChart({ data }) {
  const traffic = data.traffic || {};

  const trafficData = [
    { name: 'Vehicle Count', value: traffic.vehicle_count || 0 },
    { name: 'Avg Speed (km/h)', value: traffic.average_speed || 0 },
    { name: 'Congestion Level', value: (traffic.congestion_level || 0) * 100 }
  ];

  const getCongestionColor = (level) => {
    if (level < 0.3) return '#00e676';
    if (level < 0.6) return '#ffeb3b';
    if (level < 0.8) return '#ff9800';
    return '#f44336';
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={4}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="body2" color="textSecondary">Vehicle Count</Typography>
          <Typography variant="h5" sx={{ mt: 1 }}>
            {traffic.vehicle_count || 0}
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={4}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="body2" color="textSecondary">Average Speed</Typography>
          <Typography variant="h5" sx={{ mt: 1 }}>
            {(traffic.average_speed || 0).toFixed(1)} km/h
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={4}>
        <Paper sx={{
          p: 2,
          backgroundColor: getCongestionColor(traffic.congestion_level || 0) + '20',
          borderLeft: `4px solid ${getCongestionColor(traffic.congestion_level || 0)}`
        }}>
          <Typography variant="body2" color="textSecondary">Congestion Level</Typography>
          <Typography variant="h5" sx={{ mt: 1, color: getCongestionColor(traffic.congestion_level || 0) }}>
            {((traffic.congestion_level || 0) * 100).toFixed(1)}%
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Traffic Metrics</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={trafficData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Public Transport Efficiency</Typography>
          <Box sx={{
            fontSize: '2.5rem',
            fontWeight: 'bold',
            color: '#2196F3',
            textAlign: 'center',
            py: 3
          }}>
            {((traffic.public_transport_efficiency || 0) * 100).toFixed(1)}%
          </Box>
          <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center' }}>
            System efficiency rating
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Traffic Hotspots</Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
            {traffic.hotspots && traffic.hotspots.length > 0 ? (
              traffic.hotspots.map((spot, idx) => (
                <Paper key={idx} sx={{ p: 2, bgcolor: '#fff3cd', borderLeft: '4px solid #ff9800' }}>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{spot.location}</Typography>
                  <Typography variant="caption">Severity: {spot.severity}</Typography>
                </Paper>
              ))
            ) : (
              <Typography variant="body2" color="textSecondary">No active hotspots detected</Typography>
            )}
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default TrafficChart;