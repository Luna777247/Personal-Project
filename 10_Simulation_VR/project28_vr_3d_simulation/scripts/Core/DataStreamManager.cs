using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// DataStreamManager - Real-time environmental data updates
/// Fetches weather, flood forecasts, climate data from external APIs
/// Distributes updates to interested subscribers
/// </summary>
public class DataStreamManager : MonoBehaviour
{
    #region Singleton
    private static DataStreamManager instance;
    public static DataStreamManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<DataStreamManager>();
            }
            return instance;
        }
    }
    #endregion

    [System.Serializable]
    public class APIConfiguration
    {
        public string weatherApiUrl = "https://api.weather.gov";
        public string floodApiUrl = "https://flooddisplacement.org/api";
        public string climateApiUrl = "https://data.copernicus.eu";
        public string apiKey = "";
        public float updateFrequency = 1f; // Updates per second
    }

    [SerializeField] private APIConfiguration apiConfig = new APIConfiguration();
    
    // Current environmental state
    [System.Serializable]
    public class EnvironmentalState
    {
        // Weather
        public float temperature = 20f;
        public float humidity = 60f;
        public float windSpeed = 5f;
        public float windDirection = 0f;
        public float rainfallRate = 0f;

        // Hydrological
        public float riverFlowRate = 100f;
        public float groundwater = 50f;
        public float soilMoisture = 60f;

        // Biological
        public float vegetationHealth = 75f;
        public float wildlifeDensity = 10f;

        // Timestamp
        public System.DateTime timestamp = System.DateTime.Now;
    }

    public EnvironmentalState currentEnvironment = new EnvironmentalState();
    public EnvironmentalState CurrentEnvironment { get => currentEnvironment; }

    // Data consumers
    private List<IDataConsumer> subscribers = new List<IDataConsumer>();

    // Coroutine tracking
    private Coroutine dataStreamCoroutine;
    private bool isStreaming = false;

    // Events for data updates
    public event Action<EnvironmentalState> OnEnvironmentUpdated;
    public event Action<string> OnDataError;

    public interface IDataConsumer
    {
        void UpdateEnvironmentalData(EnvironmentalState state);
    }

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
        StartDataStream();
    }

    public void StartDataStream()
    {
        if (isStreaming)
            return;

        isStreaming = true;
        dataStreamCoroutine = StartCoroutine(DataStreamLoop());
        Debug.Log("DataStreamManager: Started streaming environmental data");
    }

    public void StopDataStream()
    {
        if (!isStreaming)
            return;

        isStreaming = false;
        if (dataStreamCoroutine != null)
        {
            StopCoroutine(dataStreamCoroutine);
        }
        Debug.Log("DataStreamManager: Stopped streaming environmental data");
    }

    private IEnumerator DataStreamLoop()
    {
        while (isStreaming)
        {
            // Fetch data from APIs
            yield return StartCoroutine(FetchEnvironmentalData());

            // Distribute to subscribers
            UpdateSubscribers();

            // Wait for next update
            float waitTime = 1f / apiConfig.updateFrequency;
            yield return new WaitForSeconds(waitTime);
        }
    }

    private IEnumerator FetchEnvironmentalData()
    {
        // Fetch weather data
        yield return StartCoroutine(FetchWeatherData());

        // Fetch flood forecast data
        yield return StartCoroutine(FetchFloodForecasts());

        // Fetch climate data
        yield return StartCoroutine(FetchClimateData());
    }

    private IEnumerator FetchWeatherData()
    {
        // Simulate API call to fetch weather
        // In production, use UnityWebRequest:
        // 
        // UnityWebRequest request = UnityWebRequest.Get(apiConfig.weatherApiUrl);
        // yield return request.SendWebRequest();
        // 
        // if (request.result == UnityWebRequest.Result.Success)
        // {
        //     string json = request.downloadHandler.text;
        //     // Parse JSON and update currentEnvironment
        // }

        // Simulated weather variation
        currentEnvironment.temperature += UnityEngine.Random.Range(-0.5f, 0.5f);
        currentEnvironment.temperature = Mathf.Clamp(currentEnvironment.temperature, -10f, 40f);
        
        currentEnvironment.humidity += UnityEngine.Random.Range(-2f, 2f);
        currentEnvironment.humidity = Mathf.Clamp01(currentEnvironment.humidity / 100f) * 100f;

        yield return null;
    }

    private IEnumerator FetchFloodForecasts()
    {
        // Simulate API call to fetch flood data
        // In production:
        // UnityWebRequest request = UnityWebRequest.Get(apiConfig.floodApiUrl);
        // yield return request.SendWebRequest();

        // Simulated flood data variation
        currentEnvironment.rainfallRate += UnityEngine.Random.Range(-0.2f, 0.3f);
        currentEnvironment.rainfallRate = Mathf.Max(0, currentEnvironment.rainfallRate);

        currentEnvironment.riverFlowRate += UnityEngine.Random.Range(-5f, 10f);
        currentEnvironment.riverFlowRate = Mathf.Max(50f, currentEnvironment.riverFlowRate);

        yield return null;
    }

    private IEnumerator FetchClimateData()
    {
        // Simulate API call to fetch climate data
        // In production:
        // UnityWebRequest request = UnityWebRequest.Get(apiConfig.climateApiUrl);
        // yield return request.SendWebRequest();

        // Simulated climate variation
        currentEnvironment.vegetationHealth += UnityEngine.Random.Range(-1f, 1f);
        currentEnvironment.vegetationHealth = Mathf.Clamp01(currentEnvironment.vegetationHealth / 100f) * 100f;

        yield return null;
    }

    private void UpdateSubscribers()
    {
        currentEnvironment.timestamp = System.DateTime.Now;

        foreach (var subscriber in subscribers)
        {
            subscriber.UpdateEnvironmentalData(currentEnvironment);
        }

        OnEnvironmentUpdated?.Invoke(currentEnvironment);
    }

    public void Subscribe(IDataConsumer consumer)
    {
        if (!subscribers.Contains(consumer))
        {
            subscribers.Add(consumer);
            Debug.Log($"DataStreamManager: Subscriber registered: {consumer.GetType().Name}");
        }
    }

    public void Unsubscribe(IDataConsumer consumer)
    {
        subscribers.Remove(consumer);
        Debug.Log($"DataStreamManager: Subscriber removed: {consumer.GetType().Name}");
    }

    public void SetUpdateFrequency(float frequency)
    {
        apiConfig.updateFrequency = frequency;
    }

    public void SetWeatherApiUrl(string url)
    {
        apiConfig.weatherApiUrl = url;
    }

    public void SetFloodApiUrl(string url)
    {
        apiConfig.floodApiUrl = url;
    }

    public void SetApiKey(string key)
    {
        apiConfig.apiKey = key;
    }

    public EnvironmentalState GetCurrentEnvironment()
    {
        return currentEnvironment;
    }

    private void OnDestroy()
    {
        StopDataStream();
    }
}
