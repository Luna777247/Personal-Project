# Quick Start Guide - Project 28 VR/3D Simulation

Get up and running with the VR/3D Simulation project in 15 minutes!

## 📋 Prerequisites

### Option A: Unity Development (Recommended for 3D scenes)
- **Unity Hub** (free) - [Download](https://unity.com/download)
- **Unity 2022 LTS** - ~50GB
- **Visual Studio Code or Rider** - for C# editing
- **Git** - version control

### Option B: WebXR (For browser-based development)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **npm or yarn** - package manager
- **VS Code** - code editor
- **Chrome/Edge browser** - with WebXR support

### Option C: Blender (For 3D modeling)
- **Blender 4.0+** - [Download](https://www.blender.org/)
- **Substance Painter** (optional) - for texturing

## 🚀 Quick Start Paths

### Path 1: Unity Desktop VR Development (20 minutes)

```bash
# 1. Clone or download project
git clone <repository-url>
cd project28_vr_3d_simulation

# 2. Open in Unity Hub
# - Click "Open project"
# - Navigate to project folder
# - Unity will load the project (takes 2-5 minutes first time)

# 3. Open the first scene
# In Project window: Assets > Scenes > WasteManagement > WasteProcessing.unity

# 4. Press Play in Unity Editor
# You should see the waste processing facility scene

# 5. Test interactions:
# - Move mouse to control view
# - WASD to move around
# - E to interact with objects
# - Space to teleport
```

**What you'll see:**
- 3D waste processing facility
- Interactive sorting mechanisms
- Real-time contamination detection
- Performance metrics display

**Next steps:**
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for codebase structure
- Read [SCENE_DESIGN.md](SCENE_DESIGN.md) to understand scenes
- Read [VR_INTERACTION.md](VR_INTERACTION.md) to customize controls

---

### Path 2: WebXR Browser VR (10 minutes)

```bash
# 1. Navigate to WebXR folder
cd webxr

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
# Output: http://localhost:5173 (or similar)

# 4. Open in browser
# - Open Chrome/Edge
# - Go to http://localhost:5173
# - Click "Enter VR" button

# 5. Test with your VR device:
# - Desktop: Use mouse + keyboard
# - Mobile VR: Use Cardboard VR controller
# - Desktop VR: Connect Meta Quest to PC
```

**What you'll see:**
- Browser-based 3D scene
- Hand controller visualization
- Gaze-based interaction cursor
- Real-time data streaming

**Device compatibility:**
- ✅ Meta Quest (via AirLink or browser)
- ✅ HTC Vive (browser WebXR)
- ✅ Mobile phone + Cardboard VR
- ✅ Desktop (emulated with mouse/keyboard)

**Development cycle:**
```bash
# Edit files
npm run dev  # Auto-reloads browser

# Build for production
npm run build

# Deploy
npm run deploy
```

---

### Path 3: Blender 3D Modeling (15 minutes)

```bash
# 1. Open Blender
blender blender_models/WasteProcessing.blend

# 2. Explore the scene
# - Use middle-mouse button to rotate view
# - Scroll to zoom
# - Shift+middle-mouse to pan

# 3. Edit existing objects
# - Select object with left-click
# - Press Tab to enter Edit Mode
# - Modify vertices/faces

# 4. Create new object
# Shift+A > Mesh > Select mesh type

# 5. Export to glTF (for use in Unity/WebXR)
# File > Export > glTF 2.0 (.glb/.gltf)
```

**Best practices:**
- Always export as glTF 2.0 (.glb) - compatible with both Unity & WebXR
- Use materials, not textures, for better performance
- Keep polygon count under 100k per model
- Test on target device before optimizing

---

## 🎮 First Interaction Test

### Desktop (No VR Headset Required)

**In Unity Editor:**
```
WASD         = Move around
Mouse Look   = Look around (hold right-click)
E            = Interact with nearby objects
Space        = Jump/Teleport
1-3          = Switch between scenarios
Esc          = Pause menu
```

**In WebXR Browser:**
```
Mouse        = Look around (hold left-click)
WASD         = Move
Click Button = Interact
F            = Enter fullscreen (VR mode)
```

### Mobile VR (Cardboard)

1. Place your phone in Cardboard headset
2. Go to `http://<your-computer-ip>:5173` on your phone
3. Click "Enter VR"
4. Look at objects (gaze selection)
5. Tap screen to interact

### Desktop VR (Meta Quest 3)

1. Connect Meta Quest via AirLink or cable
2. Open browser on Quest
3. Go to `localhost:5173` (or network URL)
4. Use hand controllers to interact

---

## 📁 Project Structure Overview

```
project28_vr_3d_simulation/
├── scenes/          ← Unity scenes (3 main scenarios)
├── scripts/         ← C# game logic
├── assets/          ← 3D models, textures, audio
├── webxr/           ← Browser-based VR
├── blender_models/  ← Source .blend files
└── docs/            ← Documentation
```

### Important Files to Know

| File | Purpose | Edit with |
|------|---------|-----------|
| `scenes/WasteManagement/WasteProcessing.unity` | Main waste facility | Unity Editor |
| `scenes/FloodRisk/FloodSimulation.unity` | Flood simulation | Unity Editor |
| `scenes/ForestEcosystem/ForestBiodiversity.unity` | Forest ecosystem | Unity Editor |
| `scripts/Core/VRInteractionManager.cs` | Hand/gaze controls | VS Code or Rider |
| `webxr/js/app.js` | WebXR main app | VS Code |
| `blender_models/WasteProcessing.blend` | 3D models | Blender |

---

## 🔧 Common Tasks

### Task 1: Add a New Interactive Object

**In Unity:**

```csharp
// 1. Create new C# script in scripts/Interaction/
// File: MyInteractiveObject.cs

using UnityEngine;

public class MyInteractiveObject : MonoBehaviour
{
    public void OnInteract()
    {
        Debug.Log("Object interacted!");
        // Add your logic here
    }
}

// 2. In Unity Editor:
// - Create a 3D cube or import model
// - Add MyInteractiveObject script
// - Add Collider component
// - Connect to VRInteractionManager

// 3. Test: Press E near object in game
```

### Task 2: Add Real-time Data

**In WebXR:**

```javascript
// File: webxr/js/dataSync.js

async function fetchEnvironmentalData() {
    try {
        const response = await fetch('http://api.example.com/weather');
        const data = await response.json();
        
        // Update scene based on data
        updateWeather(data.temperature, data.humidity);
        updateFloodLevel(data.rainfall);
        
    } catch (error) {
        console.error('Failed to fetch data:', error);
    }
}

// Call every 5 seconds
setInterval(fetchEnvironmentalData, 5000);
```

### Task 3: Create New Scene in Unity

```
1. File > New Scene
2. Name: YourSceneName.unity
3. Save in: Assets/Scenes/YourFolder/
4. Add 3D models (GameObject > 3D Object)
5. Add lights (GameObject > Light)
6. Add VR interaction manager (drag prefab)
7. Test by pressing Play
8. Save Ctrl+S
```

---

## 🐛 Troubleshooting

### Problem: "Unity won't load project"
**Solution:**
```bash
# 1. Delete Library folder (cache)
rm -r project28_vr_3d_simulation/Library

# 2. Delete .vscode folder
rm -r project28_vr_3d_simulation/.vscode

# 3. Re-open project in Unity Hub
# This will regenerate cache
```

### Problem: "WebXR not working in browser"
**Solution:**
```bash
# Check if running on localhost or HTTPS
# WebXR requires secure context!

# For local development:
npm run dev  # Should be http://localhost:5173

# For remote access:
# Use HTTPS or localhost only

# Test in Chrome DevTools:
# Console > navigator.xr  # Should exist
```

### Problem: "Assets not loading"
**Solution:**
```
1. Check Assets folder exists and is not empty
2. In Unity: Assets > Reimport All
3. Check Console for error messages
4. Verify file paths use forward slashes /
```

### Problem: "Low FPS on VR headset"
**Solution:**
```
1. Check draw calls: Stats window in Game view
2. Reduce polygon count: Optimize models in Blender
3. Disable shadows on non-essential objects
4. Use LOD (Level of Detail) system
5. Profile: Window > Analysis > Profiler
```

---

## 📚 Documentation Map

| Document | For | Reading Time |
|----------|-----|--------------|
| [README.md](../README.md) | Project overview | 15 min |
| **QUICKSTART.md** | Getting started | 10 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Code structure | 20 min |
| [SCENE_DESIGN.md](SCENE_DESIGN.md) | Scene creation | 15 min |
| [VR_INTERACTION.md](VR_INTERACTION.md) | Controls & input | 10 min |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Building & shipping | 15 min |

---

## 🎓 Learning Resources

### Unity & C# Development
- [Unity Learn](https://learn.unity.com/) - Free tutorials
- [Unity Manual](https://docs.unity3d.com/Manual/) - Official docs
- [Brackeys YouTube](https://www.youtube.com/@Brackeys) - Game dev tutorials

### VR Development
- [Meta Quest Developer Documentation](https://developer.oculus.com/)
- [OpenXR Best Practices](https://www.khronos.org/openxr/)
- [XR Interaction Toolkit Samples](https://github.com/Unity-Technologies/XR-Interaction-Toolkit-Examples)

### WebXR Development
- [Babylon.js Playground](https://playground.babylonjs.com/) - Interactive examples
- [Three.js Examples](https://threejs.org/examples/) - WebXR examples
- [WebXR Samples](https://immersive-web.github.io/webxr-samples/)

### 3D Modeling
- [Blender Beginner Tutorial Series](https://www.blender.org/support/tutorials/)
- [Substance Painter Academy](https://www.substance3d.com/academy/)
- [Poly Haven](https://polyhaven.com/) - Free 3D assets

### Educational Game Design
- [Game Based Learning Research](https://gblprinciples.org/)
- [Learning Sciences Books](https://mitpress.mit.edu/search-books?f=Series:The%20MIT%20Press%20Essential%20Knowledge%20series)

---

## ⚡ Pro Tips

1. **Always backup before major changes**
   ```bash
   git add .
   git commit -m "Before major refactor"
   ```

2. **Use prefabs for reusable objects**
   - Create once, use many times
   - Edit prefab = update all instances

3. **Enable VCS (Version Control) early**
   - Edit > Project Settings > Editor > Collaborate
   - Better than manual backups

4. **Test on actual VR device early**
   - Emulation != real VR
   - Latency, field of view are different

5. **Profile regularly**
   - Window > Analysis > Profiler
   - Catch performance issues early

6. **Document your scenes**
   - Add comments in C# code
   - Create scene diagrams
   - Keep design specs updated

---

## 🚀 Next Steps

### After your first successful run:

1. **Explore the code**
   - Read [ARCHITECTURE.md](ARCHITECTURE.md)
   - Study interaction scripts

2. **Create your first scenario variant**
   - Duplicate existing scene
   - Modify objects and materials
   - Add your own interactive elements

3. **Deploy to VR**
   - Read [DEPLOYMENT.md](DEPLOYMENT.md)
   - Build for Meta Quest or WebXR
   - Test on actual device

4. **Integrate real-world data**
   - Connect to NOAA API for weather
   - Stream live flood forecast data
   - Visualize real satellite imagery

5. **Optimize for mobile**
   - Run on Android phone
   - Use Cardboard VR
   - Target 60 FPS on mobile

---

## 🆘 Need Help?

### Check these resources in order:

1. **Search documentation**: `Ctrl+F` in this file
2. **Read related docs**: [ARCHITECTURE.md](ARCHITECTURE.md), [SCENE_DESIGN.md](SCENE_DESIGN.md)
3. **Unity documentation**: https://docs.unity3d.com/
4. **WebXR samples**: https://immersive-web.github.io/webxr-samples/
5. **Create GitHub issue**: Describe problem + error message

---

**Ready to create?** Open your preferred development environment and jump in! 🎮🌍

---

**Last Updated**: 2024  
**For**: Project 28 VR/3D Simulation  
**Author**: AI Assistant  
