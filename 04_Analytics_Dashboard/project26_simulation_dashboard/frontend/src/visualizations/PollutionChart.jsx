import React from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

function PollutionChart({ data }) {
  const pollution = data.pollution || {};

  const pollutantData = [
    { name: 'PM2.5', value: pollution.pm25 || 0 },
    { name: 'PM10', value: pollution.pm10 || 0 },
    { name: 'NO2', value: pollution.no2 || 0 },
    { name: 'SO2', value: pollution.so2 || 0 },
  ];

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];

  const getAQIColor = (aqi) => {
    if (aqi < 50) return '#00e676';
    if (aqi < 100) return '#ffeb3b';
    if (aqi < 150) return '#ff9800';
    if (aqi < 200) return '#f44336';
    if (aqi < 300) return '#9c27b0';
    return '#4a0080';
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Air Quality Index</Typography>
          <Box sx={{
            fontSize: '3rem',
            fontWeight: 'bold',
            color: getAQIColor(pollution.air_quality_index || 0),
            textAlign: 'center'
          }}>
            {(pollution.air_quality_index || 0).toFixed(1)}
          </Box>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Pollutants Breakdown</Typography>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={pollutantData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value.toFixed(1)}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {COLORS.map((color, index) => (
                  <Cell key={`cell-${index}`} fill={color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Detailed Pollutant Levels</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={pollutantData}>
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
          <Typography variant="h6" gutterBottom>Emission Sources</Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            {[
              { label: 'CO (Carbon Monoxide)', value: pollution.co },
              { label: 'O3 (Ozone)', value: pollution.o3 },
              { label: 'SO2 (Sulfur Dioxide)', value: pollution.so2 },
            ].map((item, idx) => (
              <Paper key={idx} sx={{ p: 1.5, flex: 1, minWidth: 150 }}>
                <Typography variant="body2" color="textSecondary">{item.label}</Typography>
                <Typography variant="h6">{(item.value || 0).toFixed(2)}</Typography>
              </Paper>
            ))}
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default PollutionChart;