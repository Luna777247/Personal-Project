using UnityEngine;
using UnityEngine.XR;
using System;

/// <summary>
/// HandTracker - Handles hand pose detection and gesture recognition
/// Tracks individual finger positions and detects common hand gestures
/// </summary>
public class HandTracker : MonoBehaviour
{
    private XRNode handNode;
    private InputDevice handDevice;
    
    [SerializeField] private Transform[] fingerTransforms = new Transform[5];
    
    // Hand position and rotation
    private Vector3 palmPosition;
    private Quaternion palmRotation;
    
    public Vector3 PalmPosition { get => palmPosition; }
    public Quaternion PalmRotation { get => palmRotation; }
    
    // Current hand state
    private HandPose currentPose = HandPose.Open;
    public HandPose CurrentPose { get => currentPose; }
    
    // Gesture detection thresholds
    [SerializeField] private float pinchThreshold = 0.1f;
    [SerializeField] private float grabThreshold = 0.5f;
    
    // Events
    public event Action<HandPose> OnPoseChanged;
    public event Action OnPinchStart;
    public event Action OnPinchEnd;
    
    private bool isPinching = false;
    private HandPose previousPose = HandPose.Open;

    public void Initialize(XRNode node)
    {
        handNode = node;
        
        // List all connected input devices
        InputDevices.GetDevicesWithCharacteristics(
            InputDeviceCharacteristics.HeldInHand |
            InputDeviceCharacteristics.TrackedDevice,
            System.Collections.Generic.ListPool<InputDevice>.Get(out var devices)
        );

        foreach (var device in devices)
        {
            if (device.characteristics.HasFlag(InputDeviceCharacteristics.Left) && node == XRNode.LeftHand)
            {
                handDevice = device;
                break;
            }
            else if (device.characteristics.HasFlag(InputDeviceCharacteristics.Right) && node == XRNode.RightHand)
            {
                handDevice = device;
                break;
            }
        }

        Debug.Log($"Initialized HandTracker for {node}: {handDevice.name}");
    }

    private void Update()
    {
        if (!handDevice.isValid)
            return;

        UpdateHandPosition();
        UpdateFingerPositions();
        DetectHandPose();
        DetectGestures();
    }

    private void UpdateHandPosition()
    {
        if (handDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 pos))
        {
            palmPosition = pos;
        }

        if (handDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rot))
        {
            palmRotation = rot;
        }
    }

    private void UpdateFingerPositions()
    {
        // Update individual finger joint positions
        // This would require accessing XR joint tracking or hand skeletal data
        // Implementation depends on XR plugin (Meta XR SDK, OpenXR, etc.)
        
        // Placeholder - actual implementation would use:
        // - XRHand.GetJoint() for OpenXR
        // - OVRSkeleton for Meta Quest
        // - Leap Motion SDK for Leap Motion devices
    }

    private void DetectHandPose()
    {
        HandPose newPose = ClassifyHandPose();
        
        if (newPose != currentPose)
        {
            currentPose = newPose;
            OnPoseChanged?.Invoke(newPose);
        }
        
        previousPose = currentPose;
    }

    private HandPose ClassifyHandPose()
    {
        // Get hand features
        bool isPinched = IsHandPinched();
        bool isGrabbing = IsHandGrabbing();
        bool isPointing = IsHandPointing();
        
        // Classify based on finger positions
        if (isPinched && !isGrabbing)
            return HandPose.Pinch;
        else if (isGrabbing)
            return HandPose.Grab;
        else if (isPointing)
            return HandPose.Point;
        else
            return HandPose.Open;
    }

    private bool IsHandPinched()
    {
        // Check distance between thumb and index finger
        // If distance < pinchThreshold, hand is pinched
        
        // Simplified check - in real implementation would use 3D finger positions
        if (handDevice.TryGetFeatureValue(CommonUsages.trigger, out float triggerValue))
        {
            return triggerValue > 0.8f; // Virtual pinch using trigger
        }
        
        return false;
    }

    private bool IsHandGrabbing()
    {
        // Check if all fingers are curled (grabbing position)
        if (handDevice.TryGetFeatureValue(CommonUsages.grip, out float gripValue))
        {
            return gripValue > grabThreshold;
        }
        
        return false;
    }

    private bool IsHandPointing()
    {
        // Check if only index finger is extended
        // Would compare index finger extension vs other fingers
        
        if (handDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 touchpadValue))
        {
            return touchpadValue.magnitude > 0.5f;
        }
        
        return false;
    }

    private void DetectGestures()
    {
        bool currentlyPinching = currentPose == HandPose.Pinch;
        
        if (currentlyPinching && !isPinching)
        {
            // Pinch started
            isPinching = true;
            OnPinchStart?.Invoke();
        }
        else if (!currentlyPinching && isPinching)
        {
            // Pinch ended
            isPinching = false;
            OnPinchEnd?.Invoke();
        }
    }

    public void GetFingerPositions(out Vector3[] positions)
    {
        positions = new Vector3[5];
        for (int i = 0; i < fingerTransforms.Length; i++)
        {
            if (fingerTransforms[i] != null)
                positions[i] = fingerTransforms[i].position;
        }
    }

    public bool TryGetPalmPosition(out Vector3 position)
    {
        position = palmPosition;
        return handDevice.isValid;
    }

    public bool TryGetPalmRotation(out Quaternion rotation)
    {
        rotation = palmRotation;
        return handDevice.isValid;
    }
}
