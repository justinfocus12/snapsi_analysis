import cdsapi
from calendar import monthrange
from os.path import join

client = cdsapi.Client()

for year in range(1979, 2020)[:1]:
    for month in [1,2,3,12]:
        first_weekday,days_in_month = monthrange(year,month)
        print(f'About to request year {year}, month {month}')
        target_filename = join(
            '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5',
            f't2m_{year:04}-{month:02}.nc'
            )
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": [f'{year:04}'],
            "month": [f'{month:02}'],
            "day": [f'{day:02}' for day in range(1,days_in_month+1)],
            'grid': [1.0,1.0],
            "time": [f'{hour:02}:00' for hour in [0,6,12,18]],
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        client.retrieve(dataset, request, target=target_filename)

