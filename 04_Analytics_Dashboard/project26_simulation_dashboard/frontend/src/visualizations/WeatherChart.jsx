import React from 'react';
import { Grid, Paper, Typography, Box, LinearProgress } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function WeatherChart({ data }) {
  const weather = data.weather || {};

  const getTemperatureColor = (temp) => {
    if (temp < 0) return '#0066ff';
    if (temp < 15) return '#00aa00';
    if (temp < 25) return '#ffaa00';
    if (temp < 35) return '#ff6600';
    return '#ff0000';
  };

  const weatherConditions = [
    { label: 'Temperature', value: weather.temperature, unit: '°C', color: getTemperatureColor(weather.temperature) },
    { label: 'Humidity', value: weather.humidity, unit: '%', color: '#2196F3' },
    { label: 'Pressure', value: weather.pressure, unit: 'hPa', color: '#9C27B0' },
  ];

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {weatherConditions.map((condition, idx) => (
            <Grid item xs={12} sm={4} key={idx}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                  {condition.label}
                </Typography>
                <Box sx={{
                  fontSize: '2rem',
                  fontWeight: 'bold',
                  color: condition.color,
                  mb: 1
                }}>
                  {(condition.value || 0).toFixed(1)} {condition.unit}
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Wind Conditions</Typography>
          <Box sx={{ py: 2 }}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 0.5 }}>Wind Speed: {(weather.wind_speed || 0).toFixed(1)} m/s</Typography>
              <LinearProgress variant="determinate" value={Math.min((weather.wind_speed || 0) * 10, 100)} />
            </Box>
            <Box>
              <Typography variant="body2" sx={{ mb: 0.5 }}>Wind Direction: {(weather.wind_direction || 0).toFixed(0)}°</Typography>
              <Box sx={{
                width: 100,
                height: 100,
                margin: '0 auto',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Box sx={{
                  width: 80,
                  height: 80,
                  border: '2px solid #ccc',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative'
                }}>
                  <Box sx={{
                    width: 2,
                    height: 40,
                    backgroundColor: '#2196F3',
                    transform: `rotate(${weather.wind_direction || 0}deg)`,
                    transformOrigin: 'center'
                  }} />
                  <Typography variant="caption" sx={{ position: 'absolute', bottom: 5 }}>N</Typography>
                </Box>
              </Box>
            </Box>
          </Box>
        </Paper>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Precipitation</Typography>
          <Box sx={{
            fontSize: '2.5rem',
            fontWeight: 'bold',
            color: '#2196F3',
            textAlign: 'center',
            py: 3
          }}>
            {(weather.precipitation || 0).toFixed(1)} mm
          </Box>
          {weather.precipitation > 10 ? (
            <Typography variant="body2" color="error" sx={{ textAlign: 'center' }}>
              ⚠️ Heavy precipitation expected
            </Typography>
          ) : (
            <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center' }}>
              Current precipitation level
            </Typography>
          )}
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Weather Summary</Typography>
          <Grid container spacing={1}>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2"><strong>Temperature:</strong> {(weather.temperature || 0).toFixed(1)}°C</Typography>
              <Typography variant="body2"><strong>Humidity:</strong> {(weather.humidity || 0).toFixed(1)}%</Typography>
              <Typography variant="body2"><strong>Pressure:</strong> {(weather.pressure || 0).toFixed(1)} hPa</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2"><strong>Wind Speed:</strong> {(weather.wind_speed || 0).toFixed(1)} m/s</Typography>
              <Typography variant="body2"><strong>Wind Direction:</strong> {(weather.wind_direction || 0).toFixed(0)}°</Typography>
              <Typography variant="body2"><strong>Precipitation:</strong> {(weather.precipitation || 0).toFixed(1)} mm</Typography>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default WeatherChart;