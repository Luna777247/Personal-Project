import React, { useState, useEffect } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
  Chip
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Traffic as TrafficIcon,
  Cloud as WeatherIcon,
  Nature as AgricultureIcon,
  Factory as PollutionIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon
} from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';
import { simulationAPI } from '../services/api';

const drawerWidth = 280;

const Sidebar = () => {
  const [simulations, setSimulations] = useState([]);
  const location = useLocation();

  useEffect(() => {
    const fetchSimulations = async () => {
      try {
        const data = await simulationAPI.getSimulations();
        setSimulations(data.simulations || []);
      } catch (error) {
        console.error('Failed to fetch simulations:', error);
      }
    };

    fetchSimulations();
    // Refresh every 5 seconds
    const interval = setInterval(fetchSimulations, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'completed': return 'info';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <PlayIcon />;
      case 'paused': return <PauseIcon />;
      case 'completed': return <StopIcon />;
      default: return null;
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'traffic': return <TrafficIcon />;
      case 'climate': return <WeatherIcon />;
      case 'agriculture': return <AgricultureIcon />;
      case 'pollution': return <PollutionIcon />;
      default: return <DashboardIcon />;
    }
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: drawerWidth,
          boxSizing: 'border-box',
          top: '64px', // Account for header height
          height: 'calc(100% - 64px)'
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Active Simulations
        </Typography>
      </Box>
      <Divider />

      <List>
        <ListItem disablePadding>
          <ListItemButton
            component={Link}
            to="/"
            selected={location.pathname === '/'}
          >
            <ListItemIcon>
              <DashboardIcon />
            </ListItemIcon>
            <ListItemText primary="Dashboard Overview" />
          </ListItemButton>
        </ListItem>

        {simulations.map((simulation) => (
          <ListItem key={simulation.id} disablePadding>
            <ListItemButton
              component={Link}
              to={`/simulation/${simulation.id}`}
              selected={location.pathname === `/simulation/${simulation.id}`}
            >
              <ListItemIcon>
                {getTypeIcon(simulation.type)}
              </ListItemIcon>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="body2" noWrap>
                  {simulation.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Cycle: {simulation.current_cycle}
                </Typography>
              </Box>
              <Box sx={{ ml: 1 }}>
                <Chip
                  size="small"
                  label={simulation.status}
                  color={getStatusColor(simulation.status)}
                  icon={getStatusIcon(simulation.status)}
                />
              </Box>
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default Sidebar;