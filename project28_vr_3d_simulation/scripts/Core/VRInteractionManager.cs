using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit;
using System;
using System.Collections.Generic;

/// <summary>
/// VRInteractionManager - Central hub for all VR user input processing
/// Handles hand tracking, gaze interaction, and input mapping
/// Coordinates with scenario managers and physics systems
/// </summary>
public class VRInteractionManager : MonoBehaviour
{
    #region Singleton Pattern
    private static VRInteractionManager instance;
    public static VRInteractionManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<VRInteractionManager>();
                if (instance == null)
                {
                    Debug.LogError("VRInteractionManager not found in scene!");
                }
            }
            return instance;
        }
    }
    #endregion

    #region Hand Tracking
    [SerializeField] private XRNode leftHandNode = XRNode.LeftHand;
    [SerializeField] private XRNode rightHandNode = XRNode.RightHand;
    
    private HandTracker leftHand;
    private HandTracker rightHand;
    
    public HandTracker LeftHand { get => leftHand; }
    public HandTracker RightHand { get => rightHand; }
    #endregion

    #region Gaze Tracking
    [SerializeField] private GazeInteraction gazeInteraction;
    public GazeInteraction GazeInteraction { get => gazeInteraction; }
    #endregion

    #region Interaction State
    public enum InteractionState
    {
        Idle,
        Hovering,
        Grabbing,
        UIInteracting,
        Disabled
    }
    
    private InteractionState currentState = InteractionState.Idle;
    public InteractionState CurrentState { get => currentState; }
    
    private GameObject hoveredObject;
    private GameObject selectedObject;
    public GameObject HoveredObject { get => hoveredObject; }
    public GameObject SelectedObject { get => selectedObject; }
    #endregion

    #region Events
    public event Action<GameObject> OnObjectHovered;
    public event Action<GameObject> OnObjectUnhovered;
    public event Action<GameObject> OnObjectSelected;
    public event Action<GameObject> OnObjectDeselected;
    public event Action<Vector3> OnGazePoint;
    public event Action<HandPose> OnGestureDetected;
    #endregion

    #region Configuration
    [SerializeField] private bool enableHandTracking = true;
    [SerializeField] private bool enableGazeInteraction = true;
    [SerializeField] private float interactionDistance = 2.0f;
    [SerializeField] private LayerMask interactableLayer;
    #endregion

    private void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
            return;
        }
        instance = this;
    }

    private void Start()
    {
        InitializeHandTracking();
        InitializeGazeTracking();
        Debug.Log("VRInteractionManager initialized successfully");
    }

    private void InitializeHandTracking()
    {
        if (!enableHandTracking) return;
        
        // Find or create hand trackers
        leftHand = GetComponentInChildren<HandTracker>();
        if (leftHand == null)
        {
            GameObject leftHandObj = new GameObject("LeftHand");
            leftHandObj.transform.SetParent(transform);
            leftHand = leftHandObj.AddComponent<HandTracker>();
            leftHand.Initialize(leftHandNode);
        }

        rightHand = GetComponentInChildren<HandTracker>();
        if (rightHand == null)
        {
            GameObject rightHandObj = new GameObject("RightHand");
            rightHandObj.transform.SetParent(transform);
            rightHand = rightHandObj.AddComponent<HandTracker>();
            rightHand.Initialize(rightHandNode);
        }

        if (leftHand != null)
        {
            leftHand.OnPoseChanged += HandleHandPoseChanged;
            leftHand.OnPinchStart += () => HandlePinchStart(leftHand);
            leftHand.OnPinchEnd += () => HandlePinchEnd(leftHand);
        }

        if (rightHand != null)
        {
            rightHand.OnPoseChanged += HandleHandPoseChanged;
            rightHand.OnPinchStart += () => HandlePinchStart(rightHand);
            rightHand.OnPinchEnd += () => HandlePinchEnd(rightHand);
        }
    }

    private void InitializeGazeTracking()
    {
        if (!enableGazeInteraction) return;
        
        if (gazeInteraction == null)
        {
            gazeInteraction = GetComponentInChildren<GazeInteraction>();
            if (gazeInteraction == null)
            {
                Debug.LogWarning("GazeInteraction component not found. Gaze interaction disabled.");
                enableGazeInteraction = false;
                return;
            }
        }

        gazeInteraction.OnGazeTarget += HandleGazeTarget;
        gazeInteraction.OnGazeDwell += HandleGazeDwell;
    }

    private void Update()
    {
        ProcessHandInput();
        ProcessGazeInput();
        UpdateInteractionState();
    }

    private void ProcessHandInput()
    {
        if (!enableHandTracking) return;

        // Cast rays from hand positions
        RaycastHit hit;
        
        if (leftHand != null)
        {
            if (Physics.Raycast(leftHand.PalmPosition, leftHand.PalmRotation * Vector3.forward, 
                out hit, interactionDistance, interactableLayer))
            {
                HandleObjectHover(hit.gameObject, leftHand);
            }
        }

        if (rightHand != null)
        {
            if (Physics.Raycast(rightHand.PalmPosition, rightHand.PalmRotation * Vector3.forward,
                out hit, interactionDistance, interactableLayer))
            {
                HandleObjectHover(hit.gameObject, rightHand);
            }
        }
    }

    private void ProcessGazeInput()
    {
        if (!enableGazeInteraction || gazeInteraction == null) return;

        OnGazePoint?.Invoke(gazeInteraction.GazeOrigin + gazeInteraction.GazeDirection * 100f);
    }

    private void HandleHandPoseChanged(HandPose newPose)
    {
        OnGestureDetected?.Invoke(newPose);
    }

    private void HandlePinchStart(HandTracker hand)
    {
        if (hoveredObject != null)
        {
            var manipulator = hoveredObject.GetComponent<ObjectManipulation>();
            if (manipulator != null)
            {
                manipulator.Grab(hand);
                currentState = InteractionState.Grabbing;
                OnObjectSelected?.Invoke(hoveredObject);
            }
        }
    }

    private void HandlePinchEnd(HandTracker hand)
    {
        if (selectedObject != null)
        {
            var manipulator = selectedObject.GetComponent<ObjectManipulation>();
            if (manipulator != null)
            {
                manipulator.Release();
                currentState = InteractionState.Idle;
                OnObjectDeselected?.Invoke(selectedObject);
            }
        }
    }

    private void HandleObjectHover(GameObject obj, HandTracker hand)
    {
        if (hoveredObject != obj)
        {
            // Unhover previous
            if (hoveredObject != null)
            {
                OnObjectUnhovered?.Invoke(hoveredObject);
            }

            // Hover new
            hoveredObject = obj;
            currentState = InteractionState.Hovering;
            OnObjectHovered?.Invoke(obj);
        }
    }

    private void HandleGazeTarget(GameObject target)
    {
        hoveredObject = target;
        currentState = InteractionState.Hovering;
        OnObjectHovered?.Invoke(target);
    }

    private void HandleGazeDwell()
    {
        if (hoveredObject != null)
        {
            currentState = InteractionState.UIInteracting;
            OnObjectSelected?.Invoke(hoveredObject);
        }
    }

    private void UpdateInteractionState()
    {
        // Update state based on object interactions
        // Could add logic for physics-based state transitions
    }

    public void SetInteractionEnabled(bool enabled)
    {
        currentState = enabled ? InteractionState.Idle : InteractionState.Disabled;
    }

    public void CancelCurrentInteraction()
    {
        if (selectedObject != null)
        {
            OnObjectDeselected?.Invoke(selectedObject);
            selectedObject = null;
        }
        currentState = InteractionState.Idle;
    }

    private void OnDestroy()
    {
        if (leftHand != null)
        {
            leftHand.OnPoseChanged -= HandleHandPoseChanged;
        }
        if (rightHand != null)
        {
            rightHand.OnPoseChanged -= HandleHandPoseChanged;
        }
        if (gazeInteraction != null)
        {
            gazeInteraction.OnGazeTarget -= HandleGazeTarget;
            gazeInteraction.OnGazeDwell -= HandleGazeDwell;
        }
    }
}

/// <summary>
/// HandPose enum for gesture recognition
/// </summary>
public enum HandPose
{
    Open,      // All fingers extended
    Pinch,     // Thumb + Index pinched
    Point,     // Index extended
    Grab,      // Fist closed
    Thumbsup,  // Thumb up
    Peace      // Peace sign
}
