using UnityEngine;
using System;

/// <summary>
/// ObjectManipulation - Physics-based object grabbing and throwing
/// Handles grabbing objects with hands, holding, and releasing them
/// Supports both hand tracking and traditional controller input
/// </summary>
public class ObjectManipulation : MonoBehaviour
{
    [Header("Grab Settings")]
    [SerializeField] private float grabDistance = 0.3f;
    [SerializeField] private float grabForceMultiplier = 1.0f;
    [SerializeField] private float throwVelocityMultiplier = 2.0f;
    [SerializeField] private bool useSpring = true;
    [SerializeField] private float springForce = 5000f;
    [SerializeField] private float springDamping = 500f;

    // Grabbed object tracking
    private GameObject grabbedObject;
    private Rigidbody grabbedRigidbody;
    private HandTracker grabHand;
    
    // Physics joint
    private ConfigurableJoint grabJoint;
    
    // Previous position for velocity calculation
    private Vector3 previousHandPosition;
    private Vector3 currentHandVelocity;

    // Events
    public event Action<GameObject> OnGrab;
    public event Action<GameObject> OnRelease;
    public event Action<GameObject, Vector3> OnThrow;

    private void Update()
    {
        if (grabbedObject != null && grabHand != null)
        {
            UpdateGrabbedObjectPosition();
            CalculateHandVelocity();
        }
    }

    public void Grab(HandTracker hand)
    {
        if (grabbedObject != null)
            Release(); // Release previous object

        grabbedObject = gameObject;
        grabHand = hand;
        
        if (grabbedRigidbody == null)
            grabbedRigidbody = grabbedObject.GetComponent<Rigidbody>();
        
        if (grabbedRigidbody != null)
        {
            // Create physics joint
            CreateGrabJoint(hand);
            grabbedRigidbody.velocity = Vector3.zero;
            grabbedRigidbody.angularVelocity = Vector3.zero;
        }

        previousHandPosition = hand.PalmPosition;
        OnGrab?.Invoke(grabbedObject);
        Debug.Log($"Grabbed object: {grabbedObject.name}");
    }

    public void Release()
    {
        if (grabbedObject == null)
            return;

        // Calculate throw velocity
        Vector3 throwVelocity = currentHandVelocity * throwVelocityMultiplier;

        if (grabbedRigidbody != null)
        {
            grabbedRigidbody.velocity = throwVelocity;
        }

        // Destroy joint
        if (grabJoint != null)
        {
            Destroy(grabJoint);
            grabJoint = null;
        }

        OnThrow?.Invoke(grabbedObject, throwVelocity);
        OnRelease?.Invoke(grabbedObject);

        Debug.Log($"Released object: {grabbedObject.name} with velocity: {throwVelocity.magnitude:F2} m/s");
        
        grabbedObject = null;
        grabHand = null;
    }

    private void CreateGrabJoint(HandTracker hand)
    {
        if (grabJoint != null)
            Destroy(grabJoint);

        grabJoint = grabbedObject.AddComponent<ConfigurableJoint>();
        
        if (useSpring)
        {
            // Spring-based joint (smooth following)
            JointDrive drive = new JointDrive
            {
                positionSpring = springForce,
                positionDamper = springDamping,
                maximumForce = float.MaxValue
            };

            grabJoint.xDrive = drive;
            grabJoint.yDrive = drive;
            grabJoint.zDrive = drive;

            // Allow rotation
            grabJoint.angularXDrive = drive;
            grabJoint.angularYZDrive = drive;
        }
        else
        {
            // Kinematic joint (immediate following)
            grabJoint.xMotion = ConfigurableJointMotion.Locked;
            grabJoint.yMotion = ConfigurableJointMotion.Locked;
            grabJoint.zMotion = ConfigurableJointMotion.Locked;
        }

        grabJoint.connectedBody = null; // Object follows hand without connected body
    }

    private void UpdateGrabbedObjectPosition()
    {
        if (grabJoint == null || grabHand == null)
            return;

        if (useSpring)
        {
            // Target position is hand position
            grabJoint.targetPosition = grabHand.PalmPosition - grabbedObject.transform.position;
        }
        else
        {
            // Kinematic - directly set position
            grabbedRigidbody.velocity = Vector3.zero;
            grabbedRigidbody.angularVelocity = Vector3.zero;
            grabbedObject.transform.position = grabHand.PalmPosition + 
                                                grabHand.PalmRotation * Vector3.forward * grabDistance;
            grabbedObject.transform.rotation = grabHand.PalmRotation;
        }
    }

    private void CalculateHandVelocity()
    {
        if (grabHand == null)
            return;

        Vector3 currentPosition = grabHand.PalmPosition;
        float deltaTime = Time.deltaTime;

        if (deltaTime > 0)
        {
            currentHandVelocity = (currentPosition - previousHandPosition) / deltaTime;
        }

        previousHandPosition = currentPosition;
    }

    /// <summary>
    /// Check if this object can be grabbed
    /// </summary>
    public bool IsGrabbable()
    {
        return grabbedRigidbody != null && !grabbedRigidbody.isKinematic;
    }

    /// <summary>
    /// Get visual feedback for grab state
    /// </summary>
    public bool IsCurrentlyGrabbed()
    {
        return grabbedObject != null;
    }

    /// <summary>
    /// Force release (for disabled interactions, etc.)
    /// </summary>
    public void ForceRelease()
    {
        Release();
    }

    private void OnDestroy()
    {
        if (grabJoint != null)
        {
            Destroy(grabJoint);
        }
    }
}
