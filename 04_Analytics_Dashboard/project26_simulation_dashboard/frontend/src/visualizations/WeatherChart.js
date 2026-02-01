import React, { useState, useEffect } from 'react';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Box, Typography, CircularProgress } from '@mui/material';
import { simulationAPI } from '../services/api';

const WeatherChart = ({ simulationId }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const maxDataPoints = 50;

  useEffect(() => {
    if (!simulationId) return;

    const fetchInitialData = async () => {
      try {
        const response = await simulationAPI.getSimulationData(simulationId, maxDataPoints);
        const formattedData = response.data.map(item => ({
          cycle: item.cycle,
          temperature: item.weather?.temperature || 0,
          humidity: item.weather?.humidity || 0,
          precipitation: item.weather?.precipitation || 0,
          windSpeed: item.weather?.wind_speed || 0,
          pressure: item.weather?.pressure || 0,
          timestamp: new Date(item.timestamp).toLocaleTimeString()
        }));
        setData(formattedData);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch weather data:', error);
        setLoading(false);
      }
    };

    fetchInitialData();

    // Listen for real-time updates
    const handleUpdate = (updateData) => {
      if (updateData.weather) {
        const newPoint = {
          cycle: updateData.cycle,
          temperature: updateData.weather.temperature || 0,
          humidity: updateData.weather.humidity || 0,
          precipitation: updateData.weather.precipitation || 0,
          windSpeed: updateData.weather.wind_speed || 0,
          pressure: updateData.weather.pressure || 0,
          timestamp: new Date(updateData.timestamp).toLocaleTimeString()
        };

        setData(prevData => {
          const updated = [...prevData, newPoint];
          return updated.slice(-maxDataPoints);
        });
      }
    };

    simulationAPI.onUpdate(handleUpdate);

    return () => {
      // Cleanup listeners
    };
  }, [simulationId]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (data.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body2" color="text.secondary">
          No weather data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '300px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="cycle"
            label={{ value: 'Simulation Cycle', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            yAxisId="temp"
            label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
          />
          <YAxis
            yAxisId="precip"
            orientation="right"
            label={{ value: 'Precipitation (mm)', angle: 90, position: 'insideRight' }}
          />
          <Tooltip
            labelFormatter={(value) => `Cycle: ${value}`}
            formatter={(value, name) => [
              value.toFixed(2),
              name === 'temperature' ? 'Temperature (°C)' :
              name === 'humidity' ? 'Humidity (%)' :
              name === 'precipitation' ? 'Precipitation (mm)' :
              name === 'windSpeed' ? 'Wind Speed (m/s)' : 'Pressure (hPa)'
            ]}
          />
          <Legend />

          {/* Temperature as line */}
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="temperature"
            stroke="#ff7300"
            strokeWidth={3}
            dot={false}
            name="Temperature"
          />

          {/* Precipitation as area */}
          <Area
            yAxisId="precip"
            type="monotone"
            dataKey="precipitation"
            fill="#8884d8"
            stroke="#8884d8"
            fillOpacity={0.6}
            name="Precipitation"
          />

          {/* Humidity as line */}
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="humidity"
            stroke="#82ca9d"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Humidity"
          />

          {/* Wind speed as line */}
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="windSpeed"
            stroke="#ffc658"
            strokeWidth={1}
            dot={false}
            name="Wind Speed"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default WeatherChart;