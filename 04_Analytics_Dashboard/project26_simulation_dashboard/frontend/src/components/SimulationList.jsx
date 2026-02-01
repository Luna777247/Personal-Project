import React from 'react';
import { List, ListItem, ListItemButton, ListItemText, Typography, Chip, Box } from '@mui/material';

function SimulationList({ simulations, selectedSimulation, onSelect }) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
        Active Simulations
      </Typography>
      <List>
        {simulations.map((sim) => (
          <ListItem key={sim.id} disablePadding>
            <ListItemButton
              selected={selectedSimulation?.id === sim.id}
              onClick={() => onSelect(sim)}
              sx={{
                backgroundColor: selectedSimulation?.id === sim.id ? '#e3f2fd' : 'transparent',
                borderRadius: 1,
                mb: 1
              }}
            >
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{sim.name}</span>
                    <Chip
                      label={sim.type}
                      size="small"
                      variant="outlined"
                      sx={{ ml: 1 }}
                    />
                  </Box>
                }
                secondary={`Cycle: ${sim.current_cycle}`}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default SimulationList;