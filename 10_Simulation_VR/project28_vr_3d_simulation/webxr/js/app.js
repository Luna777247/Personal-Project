/**
 * Project 28 - WebXR VR Simulation App
 * Main application entry point using Three.js
 * Supports Meta Quest, Desktop VR, and Cardboard VR
 */

import * as THREE from 'three';
import { XRButton } from 'three/addons/webxr/XRButton.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';

// Global state
const state = {
    scene: null,
    camera: null,
    renderer: null,
    controller: null,
    isVRActive: false,
    currentScenario: 'waste',
    metrics: {
        score: 0,
        time: 0,
        accuracy: 0
    },
    environment: {
        temperature: 20,
        rainfall: 0,
        humidity: 60
    }
};

/**
 * Initialize WebXR application
 */
async function init() {
    // Create scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x101010);

    // Create camera
    state.camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    state.camera.position.set(0, 1.6, 3);

    // Create renderer with WebXR support
    state.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });
    state.renderer.setPixelRatio(window.devicePixelRatio);
    state.renderer.setSize(window.innerWidth, window.innerHeight);
    state.renderer.xr.enabled = true;
    state.renderer.xr.setFoveation(1); // Optimize for performance

    document.body.appendChild(state.renderer.domElement);

    // Create XR button
    const vrButton = document.getElementById('vr-button');
    try {
        vrButton.innerHTML = XRButton.createButton(state.renderer);
    } catch (e) {
        console.warn('XRButton creation failed:', e);
        vrButton.textContent = 'WebXR not supported';
    }

    // Setup lighting
    setupLighting();

    // Create test scene
    createTestScene();

    // Setup event listeners
    setupEventListeners();

    // Start animation loop
    state.renderer.xr.addEventListener('sessionstart', onSessionStart);
    state.renderer.xr.addEventListener('sessionend', onSessionEnd);
    state.renderer.setAnimationLoop(animate);

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    console.log('WebXR app initialized successfully');
    hideLoadingScreen();
}

/**
 * Setup scene lighting
 */
function setupLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    state.scene.add(ambientLight);

    // Directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7);
    directionalLight.castShadow = true;
    state.scene.add(directionalLight);

    // Hemisphere light for better overall lighting
    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.4);
    state.scene.add(hemisphereLight);
}

/**
 * Create a test scene with basic objects
 */
function createTestScene() {
    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x008000 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    state.scene.add(ground);

    // Add some test objects
    // Cube
    const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);
    const cubeMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000 });
    const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
    cube.position.set(-2, 0.5, -5);
    cube.castShadow = true;
    state.scene.add(cube);

    // Sphere
    const sphereGeometry = new THREE.SphereGeometry(0.5, 32, 32);
    const sphereMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(0, 0.5, -5);
    sphere.castShadow = true;
    state.scene.add(sphere);

    // Cylinder
    const cylinderGeometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
    const cylinderMaterial = new THREE.MeshPhongMaterial({ color: 0x0000ff });
    const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);
    cylinder.position.set(2, 0.5, -5);
    cylinder.castShadow = true;
    state.scene.add(cylinder);

    // Add instruction text via canvas texture
    addInstructionalText();
}

/**
 * Add instructional text to scene
 */
function addInstructionalText() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Text
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 60px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Welcome to Project 28', canvas.width / 2, 100);
    ctx.fillText('VR/3D Simulation', canvas.width / 2, 180);

    ctx.font = '40px Arial';
    ctx.fillText('Look around and interact with objects', canvas.width / 2, 280);
    ctx.fillText('Pinch to grab, point to select', canvas.width / 2, 350);

    const texture = new THREE.CanvasTexture(canvas);
    const geometry = new THREE.PlaneGeometry(10, 5);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const textMesh = new THREE.Mesh(geometry, material);
    textMesh.position.set(0, 2, -10);
    state.scene.add(textMesh);
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // VR button interactions
    const vrButton = document.getElementById('vr-button');
    if (vrButton) {
        vrButton.addEventListener('click', () => {
            console.log('VR button clicked');
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyDown);

    // Touch events for mobile
    document.addEventListener('touchstart', handleTouchStart);
    document.addEventListener('touchend', handleTouchEnd);
}

/**
 * Handle keyboard input
 */
function handleKeyDown(event) {
    switch (event.key) {
        case 'Escape':
            exitVR();
            break;
        case 'd':
            toggleDebugPanel();
            break;
        case '1':
            switchScenario('waste');
            break;
        case '2':
            switchScenario('flood');
            break;
        case '3':
            switchScenario('ecosystem');
            break;
    }
}

/**
 * Handle touch input
 */
function handleTouchStart(event) {
    if (event.touches.length === 1) {
        // Single touch - interact
        console.log('Touch interact');
    }
}

function handleTouchEnd(event) {
    console.log('Touch end');
}

/**
 * Switch between scenarios
 */
function switchScenario(scenario) {
    state.currentScenario = scenario;
    console.log(`Switched to scenario: ${scenario}`);
    updateScenarioInfo();
}

/**
 * Update scenario information display
 */
function updateScenarioInfo() {
    const scenarios = {
        waste: {
            title: 'Waste Management',
            description: 'Sort waste items correctly'
        },
        flood: {
            title: 'Flood Simulation',
            description: 'Navigate flood risk scenarios'
        },
        ecosystem: {
            title: 'Forest Ecosystem',
            description: 'Explore biodiversity and ecosystems'
        }
    };

    const scenario = scenarios[state.currentScenario];
    document.getElementById('scenario-title').textContent = scenario.title;
    document.getElementById('scenario-description').textContent = scenario.description;
}

/**
 * Session start handler
 */
function onSessionStart() {
    state.isVRActive = true;
    console.log('VR session started');
    document.getElementById('status-text').textContent = 'VR Mode Active';
}

/**
 * Session end handler
 */
function onSessionEnd() {
    state.isVRActive = false;
    console.log('VR session ended');
    document.getElementById('status-text').textContent = 'Ready to start';
}

/**
 * Main animation loop
 */
function animate(time) {
    // Rotate test objects
    state.scene.children.forEach(child => {
        if (child instanceof THREE.Mesh && child.position.y > 0) {
            child.rotation.x += 0.005;
            child.rotation.y += 0.01;
        }
    });

    // Update metrics
    updateMetrics(time);

    // Update environment data
    updateEnvironmentDisplay();

    // Render
    state.renderer.render(state.scene, state.camera);
}

/**
 * Update and display metrics
 */
function updateMetrics(time) {
    // Simulate score increase
    state.metrics.score += Math.random() * 10;
    state.metrics.accuracy = Math.random() * 100;

    // Update UI
    document.getElementById('score').textContent = Math.floor(state.metrics.score);
    document.getElementById('accuracy').textContent = Math.floor(state.metrics.accuracy) + '%';

    // Update time
    const minutes = Math.floor(time / 60000);
    const seconds = Math.floor((time % 60000) / 1000);
    document.getElementById('time').textContent = 
        `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Update environment display with simulated data
 */
function updateEnvironmentDisplay() {
    // Simulate environmental changes
    state.environment.temperature += (Math.random() - 0.5) * 0.1;
    state.environment.rainfall += (Math.random() - 0.5) * 0.05;
    state.environment.humidity += (Math.random() - 0.5) * 0.2;

    // Clamp values
    state.environment.temperature = Math.max(-10, Math.min(40, state.environment.temperature));
    state.environment.rainfall = Math.max(0, state.environment.rainfall);
    state.environment.humidity = Math.max(0, Math.min(100, state.environment.humidity));

    // Update UI
    document.getElementById('temperature').textContent = 
        state.environment.temperature.toFixed(1) + '°C';
    document.getElementById('rainfall').textContent = 
        state.environment.rainfall.toFixed(1) + 'mm/h';
    document.getElementById('humidity').textContent = 
        state.environment.humidity.toFixed(1) + '%';
}

/**
 * Exit VR mode
 */
function exitVR() {
    if (state.renderer.xr.getSession()) {
        state.renderer.xr.getSession().end();
    }
}

/**
 * Toggle debug panel
 */
function toggleDebugPanel() {
    const panel = document.getElementById('debug-panel');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

/**
 * Handle window resize
 */
function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    state.camera.aspect = width / height;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(width, height);
}

/**
 * Hide loading screen when app is ready
 */
function hideLoadingScreen() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

/**
 * Check WebXR support
 */
async function checkWebXRSupport() {
    if (!navigator.xr) {
        console.error('WebXR not supported on this browser');
        return false;
    }

    try {
        const supported = await navigator.xr.isSessionSupported('immersive-vr');
        console.log(`WebXR immersive-vr supported: ${supported}`);
        return supported;
    } catch (e) {
        console.error('WebXR support check failed:', e);
        return false;
    }
}

// Start application when page loads
window.addEventListener('load', async () => {
    const supported = await checkWebXRSupport();
    if (!supported) {
        console.warn('WebXR not fully supported - app will run in non-VR mode');
    }
    init();
});

// Export for module usage
export { state, init, switchScenario, exitVR };
