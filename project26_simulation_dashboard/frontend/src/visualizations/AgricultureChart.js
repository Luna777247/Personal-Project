import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Box, Typography, CircularProgress, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { simulationAPI } from '../services/api';

const AgricultureChart = ({ simulationId }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [chartType, setChartType] = useState('bar');
  const maxDataPoints = 50;

  useEffect(() => {
    if (!simulationId) return;

    const fetchInitialData = async () => {
      try {
        const response = await simulationAPI.getSimulationData(simulationId, maxDataPoints);
        const formattedData = response.data.map(item => ({
          cycle: item.cycle,
          cropYield: item.agriculture?.crop_yield || 0,
          irrigationEfficiency: item.agriculture?.irrigation_efficiency || 0,
          soilMoisture: item.agriculture?.soil_moisture || 0,
          droughtIndex: item.agriculture?.drought_index || 0,
          pestPressure: item.agriculture?.pest_pressure || 0,
          timestamp: new Date(item.timestamp).toLocaleTimeString()
        }));
        setData(formattedData);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch agriculture data:', error);
        setLoading(false);
      }
    };

    fetchInitialData();

    // Listen for real-time updates
    const handleUpdate = (updateData) => {
      if (updateData.agriculture) {
        const newPoint = {
          cycle: updateData.cycle,
          cropYield: updateData.agriculture.crop_yield || 0,
          irrigationEfficiency: updateData.agriculture.irrigation_efficiency || 0,
          soilMoisture: updateData.agriculture.soil_moisture || 0,
          droughtIndex: updateData.agriculture.drought_index || 0,
          pestPressure: updateData.agriculture.pest_pressure || 0,
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

  const handleChartTypeChange = (event, newType) => {
    if (newType !== null) {
      setChartType(newType);
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
          No agriculture data available
        </Typography>
      </Box>
    );
  }

  const renderChart = () => {
    if (chartType === 'bar') {
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
              name === 'cropYield' ? 'Crop Yield (tons/ha)' :
              name === 'irrigationEfficiency' ? 'Irrigation Efficiency (%)' :
              name === 'soilMoisture' ? 'Soil Moisture (%)' :
              name === 'droughtIndex' ? 'Drought Index' : 'Pest Pressure'
            ]}
          />
          <Legend />
          <Bar dataKey="cropYield" fill="#8884d8" name="Crop Yield" />
          <Bar dataKey="irrigationEfficiency" fill="#82ca9d" name="Irrigation Efficiency" />
          <Bar dataKey="soilMoisture" fill="#ffc658" name="Soil Moisture" />
        </BarChart>
      );
    } else {
      return (
        <LineChart data={data}>
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
              name === 'cropYield' ? 'Crop Yield (tons/ha)' :
              name === 'irrigationEfficiency' ? 'Irrigation Efficiency (%)' :
              name === 'soilMoisture' ? 'Soil Moisture (%)' :
              name === 'droughtIndex' ? 'Drought Index' : 'Pest Pressure'
            ]}
          />
          <Legend />
          <Line type="monotone" dataKey="cropYield" stroke="#8884d8" strokeWidth={2} name="Crop Yield" />
          <Line type="monotone" dataKey="irrigationEfficiency" stroke="#82ca9d" strokeWidth={2} name="Irrigation Efficiency" />
          <Line type="monotone" dataKey="soilMoisture" stroke="#ffc658" strokeWidth={2} name="Soil Moisture" />
          <Line type="monotone" dataKey="droughtIndex" stroke="#ff7300" strokeWidth={2} strokeDasharray="5 5" name="Drought Index" />
          <Line type="monotone" dataKey="pestPressure" stroke="#00ff00" strokeWidth={2} strokeDasharray="3 3" name="Pest Pressure" />
        </LineChart>
      );
    }
  };

  return (
    <Box sx={{ width: '100%', height: '350px' }}>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={handleChartTypeChange}
          aria-label="chart type"
        >
          <ToggleButton value="bar" aria-label="bar chart">
            Bar Chart
          </ToggleButton>
          <ToggleButton value="line" aria-label="line chart">
            Line Chart
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </Box>
  );
};

export default AgricultureChart;