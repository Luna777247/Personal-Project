import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Box, Typography, CircularProgress, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { simulationAPI } from '../services/api';

const TrafficChart = ({ simulationId }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [chartType, setChartType] = useState('line');
  const maxDataPoints = 50;

  useEffect(() => {
    if (!simulationId) return;

    const fetchInitialData = async () => {
      try {
        const response = await simulationAPI.getSimulationData(simulationId, maxDataPoints);
        const formattedData = response.data.map(item => ({
          cycle: item.cycle,
          vehicleCount: item.traffic?.vehicle_count || 0,
          averageSpeed: item.traffic?.average_speed || 0,
          congestionLevel: item.traffic?.congestion_level || 0,
          efficiency: item.traffic?.public_transport_efficiency || 0,
          timestamp: new Date(item.timestamp).toLocaleTimeString()
        }));
        setData(formattedData);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch traffic data:', error);
        setLoading(false);
      }
    };

    fetchInitialData();

    // Listen for real-time updates
    const handleUpdate = (updateData) => {
      if (updateData.traffic) {
        const newPoint = {
          cycle: updateData.cycle,
          vehicleCount: updateData.traffic.vehicle_count || 0,
          averageSpeed: updateData.traffic.average_speed || 0,
          congestionLevel: updateData.traffic.congestion_level || 0,
          efficiency: updateData.traffic.public_transport_efficiency || 0,
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

  const handleChartTypeChange = (event, newChartType) => {
    if (newChartType !== null) {
      setChartType(newChartType);
    }
  };

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
          No traffic data available
        </Typography>
      </Box>
    );
  }

  const renderChart = () => {
    if (chartType === 'line') {
      return (
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="cycle"
            label={{ value: 'Simulation Cycle', position: 'insideBottom', offset: -5 }}
          />
          <YAxis yAxisId="left" label={{ value: 'Count/Speed', angle: -90, position: 'insideLeft' }} />
          <YAxis yAxisId="right" orientation="right" label={{ value: 'Level', angle: 90, position: 'insideRight' }} />
          <Tooltip
            labelFormatter={(value) => `Cycle: ${value}`}
            formatter={(value, name) => [
              value.toFixed(2),
              name === 'vehicleCount' ? 'Vehicle Count' :
              name === 'averageSpeed' ? 'Average Speed (km/h)' :
              name === 'congestionLevel' ? 'Congestion Level' : 'Public Transport Efficiency'
            ]}
          />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="vehicleCount"
            stroke="#8884d8"
            strokeWidth={2}
            dot={false}
            name="Vehicles"
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="averageSpeed"
            stroke="#82ca9d"
            strokeWidth={2}
            dot={false}
            name="Avg Speed"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="congestionLevel"
            stroke="#ff7300"
            strokeWidth={2}
            dot={false}
            name="Congestion"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="efficiency"
            stroke="#ffc658"
            strokeWidth={2}
            dot={false}
            name="Efficiency"
          />
        </LineChart>
      );
    } else {
      return (
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="cycle"
            label={{ value: 'Simulation Cycle', position: 'insideBottom', offset: -5 }}
          />
          <YAxis label={{ value: 'Values', angle: -90, position: 'insideLeft' }} />
          <Tooltip
            labelFormatter={(value) => `Cycle: ${value}`}
            formatter={(value, name) => [
              value.toFixed(2),
              name === 'vehicleCount' ? 'Vehicle Count' :
              name === 'averageSpeed' ? 'Average Speed (km/h)' :
              name === 'congestionLevel' ? 'Congestion Level' : 'Public Transport Efficiency'
            ]}
          />
          <Legend />
          <Bar dataKey="vehicleCount" fill="#8884d8" name="Vehicles" />
          <Bar dataKey="averageSpeed" fill="#82ca9d" name="Avg Speed" />
          <Bar dataKey="congestionLevel" fill="#ff7300" name="Congestion" />
          <Bar dataKey="efficiency" fill="#ffc658" name="Efficiency" />
        </BarChart>
      );
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={handleChartTypeChange}
          aria-label="chart type"
        >
          <ToggleButton value="line" aria-label="line chart">
            Line
          </ToggleButton>
          <ToggleButton value="bar" aria-label="bar chart">
            Bar
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      <Box sx={{ width: '100%', height: '250px' }}>
        <ResponsiveContainer width="100%" height="100%">
          {renderChart()}
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default TrafficChart;