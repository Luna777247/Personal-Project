# Climate Change Impact Simulation Documentation

## Overview
This project implements Agent-Based Models (ABM) to simulate the impacts of climate change events on communities and urban areas. The models focus on three main climate scenarios: flooding, drought, and heavy rainfall.

## Model Descriptions

### 1. Flood Impact Model (`flood_impact_model.gaml`)
**Purpose**: Simulates flooding effects on urban households.

**Key Components**:
- **Household Agents**: Represent families that can evacuate based on risk tolerance
- **Water Cells**: Represent flood-prone areas
- **Global Environment**: Manages overall water levels and rainfall

**Parameters**:
- Rainfall intensity
- Flood threshold
- Initial population

**Outputs**:
- Flood map visualization
- Statistics on flooded and evacuated households

### 2. Drought Impact Model (`drought_impact_model.gaml`)
**Purpose**: Models water scarcity effects on agricultural communities.

**Key Components**:
- **Farmer Agents**: Manage water resources and crop health
- **Water Sources**: Limited water supplies that deplete over time
- **Land Cells**: Represent agricultural areas

**Parameters**:
- Rainfall probability
- Evaporation rate
- Drought threshold
- Initial farmer population

**Outputs**:
- Farm landscape visualization
- Statistics on affected farmers and crop failures

### 3. Heavy Rain Model (`heavy_rain_model.gaml`)
**Purpose**: Simulates stormwater management during heavy rainfall events.

**Key Components**:
- **Resident Agents**: Urban population responding to flood risks
- **Drainage Systems**: Infrastructure managing water flow
- **Urban Cells**: City areas accumulating water

**Parameters**:
- Base rainfall
- Storm probability and intensity
- Drainage capacity
- Initial resident population

**Outputs**:
- Urban flood map
- Statistics on flooded areas and infrastructure overload

## Running the Models

### Prerequisites
1. Download and install GAMA Platform from https://gama-platform.github.io/
2. Ensure Java JDK is installed
3. Clone this repository

### Steps
1. Open GAMA Platform
2. File → Open → Select model file (.gaml)
3. Configure parameters in the experiment interface
4. Click "Run" to start simulation
5. Observe results in visualization displays

## Data Requirements
The models can be enhanced with real data:
- Climate data (rainfall, temperature patterns)
- Population density data
- Geographical information (elevation, drainage networks)
- Historical disaster data

## Analysis and Results
Each model provides:
- Real-time visualization of climate impacts
- Statistical tracking of affected agents
- Parameter sensitivity analysis
- Scenario comparison capabilities

## Extensions
Potential enhancements:
- Integration with GIS data
- Multi-scale modeling (neighborhood to city level)
- Policy intervention scenarios
- Economic impact assessment
- Climate adaptation strategies

## References
- GAMA Platform Documentation: https://gama-platform.github.io/
- Agent-Based Modeling literature on climate change
- SIMPLE, STAR-FARM, RIVERS methodologies