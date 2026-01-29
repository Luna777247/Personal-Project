using UnityEngine;
using System;

/// <summary>
/// GazeInteraction - Eye-gaze based interaction system
/// Allows interaction through eye-tracking and dwell activation
/// Essential for accessibility (no hands required)
/// </summary>
public class GazeInteraction : MonoBehaviour
{
    [Header("Gaze Settings")]
    [SerializeField] private float dwellTimeRequired = 1.5f;
    [SerializeField] private float gazeDistance = 100f;
    [SerializeField] private LayerMask interactableLayer;
    [SerializeField] private bool debugVisualization = false;

    // Eye tracking data
    private Vector3 gazeOrigin;
    private Vector3 gazeDirection;
    private float gazeConfidence = 0f;
    
    public Vector3 GazeOrigin { get => gazeOrigin; }
    public Vector3 GazeDirection { get => gazeDirection; }
    public float Confidence { get => gazeConfidence; }

    // Current target
    private GameObject currentGazeTarget;
    private float currentDwellTime = 0f;
    public GameObject CurrentGazeTarget { get => currentGazeTarget; }

    // Events
    public event Action<GameObject> OnGazeTarget;
    public event Action<GameObject> OnGazeLeave;
    public event Action OnGazeDwell;

    // Dwell visualization
    private GameObject dwellIndicator;
    private float dwellFillAmount = 0f;

    private void Start()
    {
        InitializeGazeTracking();
        CreateDwellIndicator();
    }

    private void InitializeGazeTracking()
    {
        // Initialize eye tracking
        // This would use platform-specific APIs:
        // - Meta Quest SDK: OVREyeGaze
        // - OpenXR: XREyes
        // - Vive: Vive eye tracking
        
        Debug.Log("Gaze tracking initialized (using head position as fallback)");
    }

    private void Update()
    {
        UpdateGazeRay();
        PerformGazeRaycast();
        UpdateDwellTimer();
        VisualizeDwell();
    }

    private void UpdateGazeRay()
    {
        // Get camera (head) position and forward direction as fallback
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            gazeOrigin = mainCamera.transform.position;
            gazeDirection = mainCamera.transform.forward;
            gazeConfidence = 0.8f; // Fallback confidence
        }

        // TODO: Replace with actual eye-tracking when available:
        // if (OVREyeGaze.Instance.EyeTrackingActive)
        // {
        //     gazeOrigin = OVREyeGaze.Instance.EyePosition;
        //     gazeDirection = OVREyeGaze.Instance.EyeForward;
        //     gazeConfidence = OVREyeGaze.Instance.Confidence;
        // }
    }

    private void PerformGazeRaycast()
    {
        RaycastHit hit;
        GameObject previousTarget = currentGazeTarget;

        if (Physics.Raycast(gazeOrigin, gazeDirection, out hit, gazeDistance, interactableLayer))
        {
            currentGazeTarget = hit.collider.gameObject;
        }
        else
        {
            currentGazeTarget = null;
        }

        // Trigger events
        if (currentGazeTarget != previousTarget)
        {
            if (previousTarget != null)
            {
                OnGazeLeave?.Invoke(previousTarget);
                currentDwellTime = 0f; // Reset dwell timer
            }

            if (currentGazeTarget != null)
            {
                OnGazeTarget?.Invoke(currentGazeTarget);
            }
        }
    }

    private void UpdateDwellTimer()
    {
        if (currentGazeTarget == null)
        {
            currentDwellTime = 0f;
            return;
        }

        currentDwellTime += Time.deltaTime;

        if (currentDwellTime >= dwellTimeRequired)
        {
            OnGazeDwell?.Invoke();
            currentDwellTime = 0f; // Reset for next dwell
        }
    }

    private void CreateDwellIndicator()
    {
        // Create UI element to show dwell progress
        if (dwellIndicator == null)
        {
            dwellIndicator = new GameObject("DwellIndicator");
            dwellIndicator.transform.SetParent(transform);
            dwellIndicator.SetActive(false);

            // TODO: Add UI Image component to show dwell progress
            // This would be a circular progress indicator
        }
    }

    private void VisualizeDwell()
    {
        if (currentGazeTarget == null)
        {
            if (dwellIndicator != null)
                dwellIndicator.SetActive(false);
            return;
        }

        if (dwellIndicator != null)
        {
            dwellIndicator.SetActive(true);
            dwellFillAmount = currentDwellTime / dwellTimeRequired;
            
            // Position indicator at gaze target
            dwellIndicator.transform.position = currentGazeTarget.transform.position;
            
            // TODO: Update UI Image fill amount
            // Image fillImage = dwellIndicator.GetComponent<Image>();
            // if (fillImage != null)
            //     fillImage.fillAmount = dwellFillAmount;
        }
    }

    private void OnDrawGizmos()
    {
        if (!debugVisualization)
            return;

        // Draw gaze ray
        Gizmos.color = Color.green;
        Gizmos.DrawRay(gazeOrigin, gazeDirection * gazeDistance);

        // Draw target highlight
        if (currentGazeTarget != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireCube(currentGazeTarget.transform.position, 
                               currentGazeTarget.GetComponent<Collider>().bounds.size);
        }
    }

    public float GetDwellProgress()
    {
        return Mathf.Clamp01(currentDwellTime / dwellTimeRequired);
    }

    public bool IsGazeActive()
    {
        return gazeConfidence > 0.5f;
    }

    public void ResetDwell()
    {
        currentDwellTime = 0f;
    }
}
