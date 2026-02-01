import ee

# Use service account for authentication
service_account = 'noted-falcon-454816-i0@noted-falcon-454816-i0.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'client_secret.json')
ee.Initialize(credentials)

print("Earth Engine is ready!")