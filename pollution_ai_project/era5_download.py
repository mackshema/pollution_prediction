import cdsapi

dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind"
    ],
    "year": [
        "2024"
    ],
    "month": [
        "01","02","03","04","05","06",
        "07","08","09","10","11","12"
    ],
    "day": [
        "01","02","03","04","05","06","07","08","09","10",
        "11","12","13","14","15","16","17","18","19","20",
        "21","22","23","24","25","26","27","28","29","30","31"
    ],
    "time": [
        "00:00","01:00","02:00","03:00","04:00","05:00",
        "06:00","07:00","08:00","09:00","10:00","11:00",
        "12:00","13:00","14:00","15:00","16:00","17:00",
        "18:00","19:00","20:00","21:00","22:00","23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [14, 77, 8, 81]  # Tamil Nadu region
}

client = cdsapi.Client()

output_file = r"C:\Users\sivas\OneDrive\ECE_climate_forcasting_hackthon\pollution_ai_project\wind_data.nc"

client.retrieve(dataset, request).download(output_file)

print("Download completed!")
print("File saved to:", output_file)