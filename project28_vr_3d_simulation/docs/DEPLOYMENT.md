# Deployment Guide - Project 28 VR/3D Simulation

## Overview

This guide covers building and deploying Project 28 to various VR platforms:
- Meta Quest 3 / 2
- HTC Vive (Desktop VR)
- PlayStation VR2
- WebXR (Browser-based)
- Android Cardboard VR

## Pre-Deployment Checklist

- [ ] All scenes tested and working
- [ ] Performance profiling completed (target FPS achieved)
- [ ] Build settings configured for target platform
- [ ] Graphics quality optimized
- [ ] Audio properly configured
- [ ] Save systems tested
- [ ] Input mapping validated on target device
- [ ] Documentation updated

---

## Platform 1: Meta Quest 3 (Android)

### Prerequisites

```bash
# Install Android SDK
# Download Meta Quest Developer Hub
# Install Meta XR SDK for Unity
```

### Build Steps

1. **Configure Project Settings**
   ```
   File > Build Settings
   - Platform: Android
   - Texture Compression: ASTC
   - Color Space: Linear (for better graphics)
   - Graphics API: OpenGLES 3
   ```

2. **Set Player Settings**
   ```
   Edit > Project Settings > Player
   - Company Name: Your Company
   - Product Name: VR Simulation
   - Bundle ID: com.company.vrsimulation
   - Minimum API Level: 29
   - Target API Level: 33+
   - Graphics: Optimized for VR
   ```

3. **Configure XR Settings**
   ```
   Edit > Project Settings > XR Plugin Management
   - Enable OpenXR
   - Select Meta Quest 3 plugin
   - Configure interaction profiles
   ```

4. **Build the APK**
   ```
   File > Build Settings > Build
   - Output: vrsimulation.apk
   ```

5. **Deploy to Device**
   ```bash
   # Connect Quest via USB
   adb install -r vrsimulation.apk
   
   # Or use Meta Quest Developer Hub
   ```

### Performance Optimization (Quest)

```csharp
// In your scenario scripts:

// Reduce particle count
particleSystem.maxParticles = 1000; // Instead of 10000

// Use LOD groups
lodGroup.SetLODs(new LOD[] {
    new LOD(0.8f, highQualityRenderer),
    new LOD(0.4f, mediumQualityRenderer),
    new LOD(0.0f, lowQualityRenderer)
});

// Optimize physics
Physics.defaultSolverIterations = 4;
Physics.defaultSolverVelocityIterations = 2;
```

### Testing on Quest 3

1. Open app from Quest home
2. Test hand tracking:
   - Pinch to grab objects
   - Point gesture for UI
   - Grab gesture for menus
3. Test gaze interaction:
   - Look at objects to select
   - Dwell 1.5 seconds to activate
4. Check performance:
   - Stats panel should show 72+ FPS

---

## Platform 2: HTC Vive (Desktop VR)

### Prerequisites

```bash
# Install SteamVR
# Install OpenVR
# Install Visual Studio 2022 with C++ support
```

### Build Steps

1. **Configure for PC VR**
   ```
   File > Build Settings
   - Platform: Standalone (Windows)
   - Architecture: x86_64
   - Graphics: DirectX 11/12
   ```

2. **Enable OpenXR**
   ```
   Edit > Project Settings > XR Plugin Management
   - Select OpenXR
   - Add Meta Quest 3 and HTC Vive profiles
   ```

3. **Build Executable**
   ```
   File > Build Settings > Build
   - Output: vrsimulation.exe
   ```

4. **Deploy and Test**
   ```bash
   # Plug in HTC Vive
   # Launch SteamVR
   # Run vrsimulation.exe
   ```

### Performance Optimization (Vive)

```csharp
// Higher quality settings for PC VR
int targetFPS = 90;

QualitySettings.SetQualityLevel(4); // High quality

// Increase particle effects
particleSystem.maxParticles = 50000;

// Enable advanced rendering
camera.allowDynamicResolution = false; // Fixed res for consistency
```

---

## Platform 3: WebXR (Browser-based)

### Setup Development Environment

```bash
cd webxr

# Install dependencies
npm install

# Start dev server
npm run dev

# Server runs at https://localhost:5173
```

### Building for Production

```bash
# Build optimized version
npm run build

# Output: dist/ folder

# Test production build locally
npm run preview
```

### Deploying to Server

```bash
# Upload dist/ folder to web server
# Ensure HTTPS is enabled (required for WebXR)

# Option 1: Vercel (recommended)
npm install -g vercel
vercel

# Option 2: Netlify
npm run build
# Drag & drop dist/ to Netlify

# Option 3: Custom server
scp -r dist/* user@example.com:/var/www/vrsimulation/
```

### Testing WebXR

**Desktop:**
```
1. Open https://localhost:5173
2. Click "Enter VR"
3. Use mouse/keyboard to interact
```

**Mobile VR (Cardboard):**
```
1. Open https://<your-ip>:5173 on smartphone
2. Place phone in Cardboard headset
3. Use Cardboard controller or touchscreen
4. Click "Enter VR"
```

**Standalone VR (Quest Browser):**
```
1. Open browser on Quest 3
2. Go to https://<your-ip>:5173
3. Click "Enter VR"
4. Use hand controllers
```

### WebXR Troubleshooting

```javascript
// Check WebXR support
if (navigator.xr) {
    console.log("WebXR supported!");
    navigator.xr.isSessionSupported('immersive-vr')
        .then(supported => {
            if (supported) {
                console.log("Immersive VR supported!");
            }
        });
} else {
    console.log("WebXR not supported on this browser");
}
```

---

## Platform 4: Android Cardboard VR

### Building APK for Cardboard

```bash
# Same as Meta Quest build but with different settings:

# Project Settings:
- Minimum API Level: 24 (Android 7.0)
- Remove Meta XR plugin
- Use Google Cardboard XR plugin

# Build to APK
# Install on Android phone with Cardboard VR headset
```

### Optimization for Mobile

```csharp
// Aggressive optimization for mobile VR

// Reduce resolution
targetDisplay.targetTexture.width = 1024;
targetDisplay.targetTexture.height = 1024;

// Minimize draw calls
StaticBatchingUtility.Combine(sceneBatches);

// Limit particle effects
particleSystem.maxParticles = 500; // Very low

// Disable post-processing
postProcessVolume.enabled = false;

// Use LOD aggressively
lodGroup.SetLODCount(4); // High to low detail versions
```

---

## Platform 5: PlayStation VR2

### Prerequisites

- PlayStation 5 development kit
- PlayStation VR2 headset
- PlayStation SDK for Unity

### Build Steps

1. **Configure for PS5**
   ```
   File > Build Settings
   - Platform: PlayStation 5
   ```

2. **Enable PS VR2**
   ```
   Edit > Project Settings > XR Plugin Management
   - Select PlayStation VR2 plugin
   ```

3. **Build Package**
   ```
   File > Build Settings > Build
   - Output: PS5 package format
   ```

4. **Deploy via Development Kit**
   - Use PlayStation DevKit to transfer package
   - Install on PS5 development console

---

## Performance Profiling

### Unity Profiler

```csharp
// Built-in profiler
Window > Analysis > Profiler

// Key metrics:
- CPU time (should be < 11ms for 90 FPS)
- GPU time (should be < 11ms for 90 FPS)
- Memory usage
- Draw calls (< 1000)
- Batch count
```

### Frame Rate Testing

```csharp
// In a test script
using UnityEngine;

public class PerformanceTester : MonoBehaviour
{
    private float avgFrameTime;
    private float[] frameTimes = new float[100];
    private int frameIndex = 0;

    void Update()
    {
        frameTimes[frameIndex] = Time.deltaTime;
        frameIndex = (frameIndex + 1) % 100;
        
        avgFrameTime = 0;
        foreach (float ft in frameTimes)
            avgFrameTime += ft;
        avgFrameTime /= 100f;

        float fps = 1f / avgFrameTime;
        Debug.Log($"Average FPS: {fps:F1}");
    }
}
```

### Optimization Checklist

- [ ] Draw calls < 1000
- [ ] Memory < 2GB (Quest), 3GB (PC)
- [ ] FPS stable (72+ Quest, 90+ PC, 60+ WebXR)
- [ ] Loading time < 10 seconds
- [ ] No stuttering during interaction
- [ ] Audio latency < 50ms
- [ ] Controller latency < 20ms

---

## CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/build.yml`:

```yaml
name: VR Build Pipeline

on:
  push:
    branches: [main, develop]

jobs:
  build-webxr:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: cd webxr && npm install && npm run build
      - uses: actions/upload-artifact@v3
        with:
          name: webxr-build
          path: webxr/dist/

  build-quest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Quest APK
        # This requires special GitHub actions for Unity
        run: |
          # Build commands here
          echo "Building for Quest..."
```

---

## Monitoring & Analytics

### Track User Engagement

```csharp
// Log important events
public class AnalyticsManager
{
    public static void LogScenarioStart(string scenarioName)
    {
        Debug.Log($"Scenario started: {scenarioName}");
        // Send to analytics service
    }

    public static void LogCompletion(string scenarioName, float timeSpent)
    {
        Debug.Log($"Scenario completed: {scenarioName} in {timeSpent}s");
    }

    public static void LogPerformance(float fps)
    {
        Debug.Log($"Average FPS: {fps}");
    }
}
```

### Remote Debugging

```csharp
// Send logs to remote server
public class RemoteLogger : MonoBehaviour
{
    private string logServerUrl = "https://your-server.com/api/logs";

    public void SendLog(string message, string level = "INFO")
    {
        StartCoroutine(SendLogToServer(message, level));
    }

    private IEnumerator SendLogToServer(string message, string level)
    {
        // Implementation using UnityWebRequest
    }
}
```

---

## Post-Deployment

### Monitoring Checklist

- [ ] App runs without crashes
- [ ] FPS consistently meets target
- [ ] Audio works correctly
- [ ] Hand tracking responsive
- [ ] Network connectivity stable (if streaming data)
- [ ] Battery drain acceptable (mobile)
- [ ] Heat management OK

### Update Process

```bash
# For WebXR
npm run build
# Upload new dist/ to server

# For Quest
# Build new APK
adb install -r new_build.apk

# For PC VR
# Build new executable
# Distribute via Steam or custom launcher
```

---

## Troubleshooting Deployment Issues

### Quest App Won't Launch

```bash
# Check logs
adb logcat | grep "your-app-name"

# Common issues:
# - API level too low
# - Permission missing
# - Corrupted APK
```

### WebXR Not Working

```javascript
// Check browser console for errors
console.error(error);

// Verify HTTPS:
// WebXR requires secure context

// Check device compatibility:
navigator.xr.isSessionSupported('immersive-vr')
```

### Low FPS

- Profile with Profiler window
- Reduce draw calls (batch more objects)
- Lower particle count
- Disable post-processing
- Use LOD groups

---

**Deployment Version**: 1.0  
**Last Updated**: 2024  
**Supported Platforms**: Meta Quest 3, HTC Vive, PlayStation VR2, WebXR, Cardboard VR
