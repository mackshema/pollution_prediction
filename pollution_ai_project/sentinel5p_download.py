import ee

# Initialize Earth Engine
ee.Initialize(project='pollutionece')

# Define Tamil Nadu region
region = ee.Geometry.Rectangle([77.0, 8.0, 81.0, 14.0])

# Load Sentinel-5P NO2 dataset
collection = (
    ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
    .select("NO2_column_number_density")
    .filterDate("2022-01-01", "2024-12-31")
    .filterBounds(region)
)

print("Total images found:", collection.size().getInfo())

# Function to extract pollution value
def extract_pollution(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000
    )
    
    return ee.Feature(None, {
        "date": image.date().format("YYYY-MM-dd"),
        "no2": stats.get("NO2_column_number_density")
    })

features = collection.map(extract_pollution)

# Export to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=ee.FeatureCollection(features),
    description="sentinel_pollution",
    fileFormat="CSV"
)

task.start()

print("Export started!")
print("Check your Google Drive in 10–20 minutes.")
