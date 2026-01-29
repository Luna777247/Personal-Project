# Architecture Document - Project 28 VR/3D Simulation

## System Overview

Project 28 is a multi-platform immersive VR experience featuring three environmental education scenarios. The architecture separates concerns across game engine, physics simulation, interaction systems, and data integration layers.

```
┌─────────────────────────────────────────────────────────────┐
│                     VR Display Layer                        │
│  (Meta Quest 3 / HTC Vive / WebXR / Cardboard VR)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼───┐   ┌─────▼────┐   ┌────▼───┐
   │ Unity  │   │ WebXR    │   │ Mobile │
   │ Desktop│   │ Babylon/ │   │ Cardboard
   │        │   │ Three.js │   │        │
   └────┬───┘   └─────┬────┘   └────┬───┘
        │              │             │
        └──────────────┼─────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Interaction Management     │
        ├──────────────────────────────┤
        │ • Hand Tracking             │
        │ • Gaze Interaction          │
        │ • Physics-based Grabbing    │
        │ • Menu Navigation           │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Scenario Managers          │
        ├──────────────────────────────┤
        │ • WasteManager              │
        │ • FloodSimulator            │
        │ • EcosystemManager          │
        │ • Data Stream Manager       │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Physics & Simulation       │
        ├──────────────────────────────┤
        │ • Water Physics (SPH)       │
        │ • Particle Effects          │
        │ • Physics Rigidbodies       │
        │ • Terrain Collision         │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  External Data Integration  │
        ├──────────────────────────────┤
        │ • NOAA Weather API          │
        │ • Flood Forecast Data       │
        │ • Climate Scenarios         │
        │ • Geospatial Data           │
        └──────────────────────────────┘
```

## Component Architecture

### 1. Core Management Systems

#### VRInteractionManager
**Location**: `scripts/Core/VRInteractionManager.cs`

**Responsibility**: Central hub for all user input processing

```csharp
public class VRInteractionManager : MonoBehaviour
{
    // Hand tracking
    public HandTracker leftHand;
    public HandTracker rightHand;
    
    // Gaze tracking
    public GazeTracker gazeTracker;
    
    // Current interaction state
    private InteractionState currentState;
    private GameObject hoveredObject;
    private GameObject selectedObject;
    
    // Events
    public event Action<GameObject> OnObjectHovered;
    public event Action<GameObject> OnObjectSelected;
    public event Action<Vector3> OnGazePoint;
}
```

**Key Methods**:
- `ProcessHandInput()` - Handle controller input
- `ProcessGazeInput()` - Process eye-tracking
- `OnGrab(Hand hand)` - Grab object with hand
- `OnRelease(Hand hand)` - Release grabbed object
- `OnUISelect()` - Select UI element

**Data Flow**:
```
Hardware Input → VRInteractionManager → Scenario Manager → GameObject
                                    ↓
                          Physics Engine / Animation
```

#### DataStreamManager
**Location**: `scripts/Core/DataStreamManager.cs`

**Responsibility**: Real-time environmental data updates

```csharp
public class DataStreamManager : MonoBehaviour
{
    // API endpoints
    private string weatherApiUrl = "https://api.weather.gov";
    private string floodApiUrl = "https://flooddisplacement.org/api";
    
    // Update frequency (Hz)
    public float updateFrequency = 1f; // 1 update per second
    
    // Current environmental state
    public EnvironmentalState currentEnvironment;
    
    // Subscribers
    private List<IDataConsumer> subscribers;
}
```

**Key Methods**:
- `FetchWeatherData()` - Get real-time weather
- `FetchFloodForecasts()` - Stream flood predictions
- `UpdateEnvironment()` - Distribute updates to subscribers
- `StreamLiveData()` - WebSocket connection for real-time data

#### UIManager
**Location**: `scripts/Core/UIManager.cs`

**Responsibility**: VR UI system (menus, HUD, panels)

```csharp
public class UIManager : MonoBehaviour
{
    // Main menu
    public VRMenu mainMenu;
    
    // In-game HUD
    public HUDElement scoreDisplay;
    public HUDElement environmentalMetrics;
    public HUDElement instructionPanel;
    
    // Interaction methods
    public void ShowMenu();
    public void HideMenu();
    public void ShowScenarioInfo(Scenario scenario);
}
```

---

### 2. Scenario Systems

#### WasteManager
**Location**: `scripts/Scenarios/WasteManager.cs`

**Responsibility**: Waste processing scenario logic

```
Waste Processing Scenario
├── InputPhase
│   ├── Unsorted waste conveyor
│   ├── Player picks items
│   └── Drops in correct bin
│
├── ProcessingPhase
│   ├── Machine processes items
│   ├── Contamination detection
│   └── Efficiency metrics
│
└── OutputPhase
    ├── Recycled material storage
    ├── Waste disposal
    └── Performance feedback
```

**Key Components**:

```csharp
public class WasteManager : MonoBehaviour
{
    // Waste types
    public enum WasteType { Plastic, Paper, Metal, Glass, Organic, Hazardous }
    
    // Bin system
    public WasteBin[] sortingBins;
    public WasteBin wasteInput;
    
    // Performance metrics
    public float sortingAccuracy { get; private set; }
    public float processingTime { get; private set; }
    public float contamination { get; private set; }
    
    // Event system
    public event Action<WasteType> OnWasteSorted;
    public event Action OnContamination;
}
```

**Interaction Flow**:
1. **Detection**: Hand enters waste conveyor area
2. **Grab**: Player grabs waste item with pinch gesture
3. **Categorize**: Item identified by color/shape
4. **Sort**: Player places in appropriate bin
5. **Feedback**: Visual/audio feedback, score update

#### FloodSimulator
**Location**: `scripts/Scenarios/FloodSimulator.cs`

**Responsibility**: Real-time water physics and flood progression

```
Flood Simulation Scenario
├── Water Physics
│   ├── SPH (Smoothed Particle Hydrodynamics)
│   ├── Velocity field
│   └── Surface tension
│
├── Terrain Interaction
│   ├── Heightmap-based collisions
│   ├── Flow direction calculation
│   └── Erosion simulation
│
├── Building Interaction
│   ├── Vulnerability assessment
│   ├── Water damage progression
│   └── Damage visualization
│
└── Multi-Scenario Support
    ├── 1-year flood
    ├── 10-year flood
    ├── 100-year flood
    └── Climate change scenario
```

**Key Components**:

```csharp
public class FloodSimulator : MonoBehaviour
{
    // Water particles (SPH)
    public FluidParticle[] waterParticles;
    
    // Terrain data
    public TerrainCollider terrain;
    public float[] heightmap;
    
    // Buildings and vulnerability
    public Building[] buildings;
    public FloodVulnerabilityMap vulnerabilityMap;
    
    // Simulation parameters
    public float rainfallIntensity; // mm/hour
    public float riverFlowRate;    // m³/s
    public float timeScale;         // 1.0 = real-time, 100.0 = 100x faster
    
    // Scenarios
    public enum FloodScenario { Year1, Year10, Year100, Climate2050, Climate2100 }
    public FloodScenario currentScenario;
}
```

**Physics Algorithm** (SPH):
```
For each particle:
  1. Find neighboring particles (within smoothing distance h)
  2. Calculate pressure gradient
  3. Calculate viscosity
  4. Apply gravity and external forces
  5. Update velocity and position
  6. Handle collisions with terrain/buildings

Update frequency: 60 Hz (real-time)
Particle count: 10,000 - 100,000 (depends on platform)
```

#### EcosystemManager
**Location**: `scripts/Scenarios/EcosystemManager.cs`

**Responsibility**: Forest ecosystem simulation with species interactions

```
Ecosystem Scenario
├── Species Management
│   ├── Trees (producers)
│   ├── Herbivores (primary consumers)
│   ├── Carnivores (secondary consumers)
│   └── Decomposers (fungi, bacteria)
│
├── Lifecycle Simulation
│   ├── Birth/reproduction
│   ├── Growth/aging
│   ├── Interaction/predation
│   └── Death/decomposition
│
├── Environmental Stressors
│   ├── Drought (rainfall reduction)
│   ├── Temperature change
│   ├── Disease/pest infestation
│   └── Human activity (logging)
│
└── Visualization
    ├── Biomass bar charts
    ├── Food web diagrams
    ├── Carbon cycle visualization
    └── Biodiversity metrics
```

**Key Components**:

```csharp
public class EcosystemManager : MonoBehaviour
{
    // Species populations
    public SpeciesPopulation[] populations;
    
    // Environmental conditions
    public EnvironmentalConditions environment;
    
    // Interaction network
    public FoodWeb foodWeb;
    
    // Metrics
    public BiodiversityMetrics biodiversity;
    public EnergyFlow energyFlow;
    
    // Simulation speed
    public float timeScale; // 1.0 = 1 day per second
}
```

**Population Dynamics** (Lotka-Volterra equations):

$$\frac{dP}{dt} = r \cdot P - c \cdot P \cdot Q$$

$$\frac{dQ}{dt} = e \cdot c \cdot P \cdot Q - d \cdot Q$$

Where:
- P = prey population
- Q = predator population
- r = prey growth rate
- c = predation rate
- e = efficiency conversion
- d = predator death rate

---

### 3. Interaction Systems

#### HandTracking
**Location**: `scripts/Interaction/HandTracking.cs`

**Responsibility**: Hand pose detection and gesture recognition

```csharp
public class HandTracking : MonoBehaviour
{
    // Hand pose detection
    public enum HandPose
    {
        Open,      // All fingers extended
        Pinch,     // Thumb + Index pinched
        Point,     // Index extended
        Grab,      // Fist closed
        Thumbsup,  // Thumb up
        Peace      // Peace sign
    }
    
    // Current hand state
    public HandPose currentPose;
    public Transform[] fingerTransforms;
    
    // Hand tracking data
    public Vector3 palmPosition;
    public Quaternion palmRotation;
    
    // Gesture callbacks
    public event Action<HandPose> OnPoseChanged;
    public event Action OnPinchStart;
    public event Action OnPinchEnd;
}
```

**Hand Gesture Flow**:
```
Raw Tracking Data
    ↓
Finger Joint Extraction
    ↓
Pose Classification
    ├── Distance metrics
    ├── Angle calculations
    └── Machine learning model
    ↓
Gesture Recognition
    ├── Temporal matching
    ├── Confidence scoring
    └── Debouncing
    ↓
Application Event
    ↓
Interaction Response
```

#### GazeInteraction
**Location**: `scripts/Interaction/GazeInteraction.cs`

**Responsibility**: Eye-tracking based interaction (accessibility feature)

```csharp
public class GazeInteraction : MonoBehaviour
{
    // Eye tracking
    public Vector3 gazeOrigin;
    public Vector3 gazeDirection;
    public float confidence; // 0-1
    
    // Dwell activation
    public float dwellTimeRequired = 1.5f; // seconds
    private float currentDwellTime;
    
    // Current target
    public GameObject currentGazeTarget;
    public event Action<GameObject> OnGazeTarget;
    public event Action OnGazeDwell; // After dwelling 1.5s
}
```

**Gaze Interaction Flow**:
```
Eye Position + Direction
    ↓
Ray Casting
    ├── Intersect UI panels
    ├── Intersect world objects
    └── Get closest object
    ↓
Hover Feedback
    ├── Visual highlight
    ├── Audio signal
    └── UI tooltip
    ↓
Dwell Detection
    ├── Track time on target
    ├── Show timer
    ↓
Activation (after 1.5s)
    ├── Trigger action
    ├── Play confirmation sound
    └── Log interaction
```

#### ObjectManipulation
**Location**: `scripts/Interaction/ObjectManipulation.cs`

**Responsibility**: Physics-based object grabbing and throwing

```csharp
public class ObjectManipulation : MonoBehaviour
{
    // Grabbed object tracking
    public GameObject grabbedObject;
    public Hand grabHand;
    
    // Physics
    public Rigidbody objectRigidbody;
    public ConfigurableJoint grabJoint;
    
    // Throw mechanics
    public float throwVelocityMultiplier = 2.0f;
    
    // Callbacks
    public event Action<GameObject> OnGrab;
    public event Action<GameObject> OnRelease;
    public event Action<GameObject, Vector3> OnThrow;
}
```

**Object Grab Physics**:
```
Grab Detection
    ├── Hand proximity check
    ├── Hand velocity < threshold
    └── Pinch gesture detected
    ↓
Joint Creation
    ├── ConfigurableJoint on object
    ├── Connected body = hand
    └── Drive constraints
    ↓
Position Tracking
    ├── Update target position
    ├── Kinematic hand tracking
    └── Haptic feedback
    ↓
Release Detection
    ├── Pinch released
    ├── Hand too far away
    └── Gesture interrupted
    ↓
Apply Velocity
    ├── Calculate hand velocity
    ├── Multiply by throw factor
    └── Impart to object rigidbody
```

---

### 4. Physics Systems

#### WaterPhysics
**Location**: `scripts/Physics/WaterPhysics.cs`

**Implementation**: Smoothed Particle Hydrodynamics (SPH)

```csharp
public class WaterPhysics : MonoBehaviour
{
    // Particle system
    public struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float pressure;
        public float density;
        public float mass;
    }
    
    public Particle[] particles;
    
    // SPH parameters
    public float h = 0.1f;          // Smoothing length
    public float restDensity = 1.0f;
    public float gasConstant = 0.0f;
    public float viscosity = 0.01f;
    public float surfaceTension = 0.01f;
    
    // Simulation constants
    public float gravity = 9.81f;
    public float deltaTime = 0.016f; // 60 Hz
}
```

**SPH Algorithm Steps**:

1. **Density Calculation**: For each particle, sum masses of neighbors weighted by kernel function
2. **Pressure Calculation**: P = k(ρ - ρ₀) where k is gas constant
3. **Force Calculation**:
   - Pressure gradient: ∇P/ρ
   - Viscosity: μ∇²v
   - Surface tension: σκ∇²C
4. **Integration**: Update velocity and position using acceleration
5. **Collision Detection**: Check against terrain and building colliders
6. **Boundary Conditions**: Pressure-constrained fluid particles

#### ParticleEffects
**Location**: `scripts/Physics/ParticleEffects.cs`

**Responsibility**: Visual particle effects for waste, water, environmental effects

```csharp
public class ParticleEffects : MonoBehaviour
{
    // Particle emitter
    public ParticleSystem waterSplash;
    public ParticleSystem dustParticles;
    public ParticleSystem smokeEmitter;
    
    // Effect triggering
    public void TriggerWaterSplash(Vector3 position, float intensity);
    public void TriggerWasteFall(WasteType waste, Vector3 position);
    public void TriggerEnvironmentalEffect(string effectType, Vector3 position);
}
```

**Visual Effects List**:
- Water splash (flood scenario)
- Waste falling particles (waste scenario)
- Dust/debris (ecosystem stress)
- Smoke/pollution (industrial waste)
- Light refraction (underwater view)

---

## Data Flow Architecture

### Real-time Data Integration Pipeline

```
┌──────────────────────┐
│ External Data Source │
│ (NOAA, Copernicus)  │
└──────────┬───────────┘
           │ HTTP/WebSocket
           ▼
┌──────────────────────┐
│ DataStreamManager    │
│ (Parse + Buffer)    │
└──────────┬───────────┘
           │ EnvironmentalState
           ▼
┌──────────────────────┐
│ Scenario Managers    │
│ (Apply to world)    │
├──────────────────────┤
│ • FloodSimulator    │
│ • EcosystemManager  │
│ • WasteManager      │
└──────────┬───────────┘
           │ GameObject updates
           ▼
┌──────────────────────┐
│ Physics Engine       │
│ (Simulate effects)   │
└──────────┬───────────┘
           │ Particle updates
           ▼
┌──────────────────────┐
│ VR Display           │
│ (Render frame)       │
└──────────────────────┘
```

### Interaction Event Flow

```
Input Device
  ├─ VR Controller (buttons, triggers, thumbstick)
  ├─ Hand Tracking (joint positions, poses)
  └─ Eye Tracker (gaze position, confidence)
            │
            ▼
VRInteractionManager
    │
    ├─ Gesture Recognition (pinch, point, grab)
    ├─ Ray Casting (what did user interact with?)
    └─ Input Validation (is action legal?)
            │
            ▼
Interaction Handlers
    ├─ ObjectManipulation (grab/throw)
    ├─ UIManager (menu selection)
    └─ Scenario Manager (task action)
            │
            ▼
Application State Update
    ├─ Physics simulation
    ├─ Score/metrics update
    └─ Audio/visual feedback
            │
            ▼
Render & Display
```

---

## Data Models

### Environmental State

```csharp
[System.Serializable]
public class EnvironmentalState
{
    // Weather
    public float temperature;      // Celsius
    public float humidity;         // 0-100%
    public float windSpeed;        // m/s
    public float windDirection;    // degrees
    public float rainfallRate;     // mm/hour
    
    // Hydrological
    public float riverFlowRate;    // m³/s
    public float groundwater;      // mm
    public float soilMoisture;     // 0-100%
    
    // Biological
    public float vegetationHealth; // 0-100%
    public float wildlifeDensity;  // creatures/km²
    
    // Timestamp
    public System.DateTime timestamp;
}
```

### Scenario Metrics

```csharp
public class ScenarioMetrics
{
    // Common metrics
    public float playerScore;
    public float completionTime;
    public int decisionsCorrect;
    public int decisionsMade;
    
    // Scenario-specific
    public Dictionary<string, float> customMetrics;
    
    // Learning outcomes
    public float knowledgeGain;      // Pre/post assessment delta
    public float engagementScore;    // Time spent, interactions
}
```

---

## Performance Optimization

### Target Performance

| Platform | FPS | Memory | Draw Calls |
|----------|-----|--------|-----------|
| Quest 3 | 72+ | <2GB | <1000 |
| PC VR | 90+ | <3GB | <1000 |
| WebXR | 60+ | <500MB | <500 |
| Mobile | 60+ | <1GB | <300 |

### Optimization Techniques

1. **Spatial Partitioning** (Quadtree/Octree)
   - Culling objects outside view frustum
   - Efficient collision detection
   - Nearby object queries

2. **Level of Detail (LOD)**
   - High-poly models for close-up view
   - Low-poly for distant objects
   - Automatic LOD switching based on distance

3. **Instancing**
   - GPU instancing for repeated objects
   - One draw call for many instances
   - Used for: trees, waste items, particles

4. **Asynchronous Loading**
   - Load scenes in background
   - Don't block main thread
   - Display loading bar

5. **Shader Optimization**
   - Custom shaders for VR (high-performance)
   - Avoid expensive calculations
   - Use compute shaders for physics

6. **Physics Optimization**
   - Use kinematic vs dynamic rigidbodies appropriately
   - Reduce particle count on lower-end hardware
   - Use layer-based collision matrices

---

## Module Dependencies

```
VRInteractionManager
  ├─ HandTracking
  ├─ GazeInteraction
  └─ UIManager

DataStreamManager
  └─ [External APIs]

Scenario Systems
  ├─ WasteManager
  │   └─ ObjectManipulation
  ├─ FloodSimulator
  │   └─ WaterPhysics
  └─ EcosystemManager
      └─ SpeciesAI

Physics
  ├─ WaterPhysics (SPH)
  ├─ ParticleEffects
  └─ TerrainCollision
```

---

## Platform-Specific Implementations

### Unity Desktop/VR
- Native C# with XR Toolkit
- Full physics and graphics
- All features supported

### WebXR (Babylon.js)
- JavaScript/TypeScript
- Subset of features (gaze + simplified physics)
- Optimized for browser

### Mobile Cardboard VR
- Android/iOS WebXR
- Ultra-optimized models
- Gaze-only interaction

---

## Extensibility Points

Developers can extend the system by:

1. **Adding New Scenarios**
   - Inherit from `Scenario` base class
   - Implement interaction handlers
   - Register with scenario manager

2. **Custom Physics**
   - Implement `IPhysicsSimulator`
   - Register in physics manager
   - Examples: cloth simulation, soft body

3. **Data Integration**
   - Implement `IDataConsumer`
   - Subscribe to `DataStreamManager`
   - React to environmental changes

4. **New Interactions**
   - Create gesture recognizer
   - Register with `VRInteractionManager`
   - Bind to application logic

---

**Architecture Version**: 1.0  
**Last Updated**: 2024  
**Supported Platforms**: Meta Quest 3, HTC Vive, WebXR, Android Cardboard  
