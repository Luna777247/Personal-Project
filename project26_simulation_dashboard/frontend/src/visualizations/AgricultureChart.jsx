import React from 'react';
import { Grid, Paper, Typography, Box, LinearProgress } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function AgricultureChart({ data }) {
  const agriculture = data.agriculture || {};

  const farmData = [
    { name: 'Crop Yield', value: agriculture.crop_yield || 0 },
    { name: 'Soil Moisture', value: agriculture.soil_moisture || 0 },
    { name: 'Irrigation Eff.', value: (agriculture.irrigation_efficiency || 0) * 100 }
  ];

  const getDroughtColor = (index) => {
    if (index < 0.3) return '#00e676';
    if (index < 0.6) return '#ffeb3b';
    if (index < 0.8) return '#ff9800';
    return '#f44336';
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>Crop Yield</Typography>
          <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 2 }}>
            {(agriculture.crop_yield || 0).toFixed(1)} units
          </Typography>
          <LinearProgress variant="determinate" value={Math.min((agriculture.crop_yield || 0) / 100 * 100, 100)} />
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{
          p: 2,
          backgroundColor: getDroughtColor(agriculture.drought_index) + '20',
          borderLeft: `4px solid ${getDroughtColor(agriculture.drought_index)}`
        }}>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>Drought Index</Typography>
          <Typography variant="h4" sx={{
            fontWeight: 'bold',
            mb: 2,
            color: getDroughtColor(agriculture.drought_index)
          }}>
            {((agriculture.drought_index || 0) * 100).toFixed(1)}%
          </Typography>
          <Typography variant="caption">
            {agriculture.drought_index < 0.3 ? 'Low drought risk' : agriculture.drought_index < 0.6 ? 'Moderate drought risk' : agriculture.drought_index < 0.8 ? 'High drought risk' : 'Severe drought conditions'}
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Soil Moisture Level</Typography>
          <Box sx={{ py: 2 }}>
            <LinearProgress
              variant="determinate"
              value={Math.min((agriculture.soil_moisture || 0), 100)}
              sx={{
                height: 20,
                borderRadius: 5,
                backgroundColor: '#e0e0e0',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: agriculture.soil_moisture > 50 ? '#4CAF50' : agriculture.soil_moisture > 30 ? '#FFC107' : '#F44336'
                }
              }}
            />
            <Typography variant="body2" sx={{ mt: 1 }}>
              {(agriculture.soil_moisture || 0).toFixed(1)}% moisture
            </Typography>
          </Box>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Irrigation Efficiency</Typography>
          <Box sx={{
            fontSize: '2.5rem',
            fontWeight: 'bold',
            color: '#4CAF50',
            textAlign: 'center',
            py: 2
          }}>
            {((agriculture.irrigation_efficiency || 0) * 100).toFixed(1)}%
          </Box>
          <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center' }}>
            System efficiency rating
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Agricultural Metrics</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={farmData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Affected Area</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Area under stress:</Typography>
                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                  {(agriculture.affected_area || 0).toFixed(1)} hectares
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min((agriculture.affected_area || 0) / 100 * 100, 100)}
                sx={{
                  backgroundColor: '#e0e0e0',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: '#FF6F00'
                  }
                }}
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default AgricultureChart;