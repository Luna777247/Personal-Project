import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Box, Typography, CircularProgress } from '@mui/material';
import { simulationAPI } from '../services/api';

const PollutionChart = ({ simulationId }) => {
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
          aqi: item.pollution?.air_quality_index || 0,
          pm25: item.pollution?.pm25 || 0,
          pm10: item.pollution?.pm10 || 0,
          no2: item.pollution?.no2 || 0,
          timestamp: new Date(item.timestamp).toLocaleTimeString()
        }));
        setData(formattedData);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch pollution data:', error);
        setLoading(false);
      }
    };

    fetchInitialData();

    // Listen for real-time updates
    const handleUpdate = (updateData) => {
      if (updateData.pollution) {
        const newPoint = {
          cycle: updateData.cycle,
          aqi: updateData.pollution.air_quality_index || 0,
          pm25: updateData.pollution.pm25 || 0,
          pm10: updateData.pollution.pm10 || 0,
          no2: updateData.pollution.no2 || 0,
          timestamp: new Date(updateData.timestamp).toLocaleTimeString()
        };

        setData(prevData => {
          const updated = [...prevData, newPoint];
          // Keep only the last maxDataPoints
          return updated.slice(-maxDataPoints);
        });
      }
    };

    simulationAPI.onUpdate(handleUpdate);

    return () => {
      // Cleanup listeners when component unmounts
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
          No pollution data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '300px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="cycle"
            label={{ value: 'Simulation Cycle', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            label={{ value: 'Pollution Level', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            labelFormatter={(value) => `Cycle: ${value}`}
            formatter={(value, name) => [
              value.toFixed(2),
              name === 'aqi' ? 'Air Quality Index' :
              name === 'pm25' ? 'PM2.5' :
              name === 'pm10' ? 'PM10' : 'NO₂'
            ]}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="aqi"
            stroke="#ff7300"
            strokeWidth={2}
            dot={false}
            name="AQI"
          />
          <Line
            type="monotone"
            dataKey="pm25"
            stroke="#8884d8"
            strokeWidth={1}
            dot={false}
            name="PM2.5"
          />
          <Line
            type="monotone"
            dataKey="pm10"
            stroke="#82ca9d"
            strokeWidth={1}
            dot={false}
            name="PM10"
          />
          <Line
            type="monotone"
            dataKey="no2"
            stroke="#ffc658"
            strokeWidth={1}
            dot={false}
            name="NO₂"
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PollutionChart;