import cdsapi
from calendar import monthrange
from os.path import join

c = cdsapi.Client()

for year in range(1979,2019):
    for month in [1,2,3]:
        first_weekday,days_in_month = monthrange(year,month)
        print(f'Beginning to download year {year}, month {month}')
        target_filename = join(
                '/gws/nopw/j04/snapsi/processed/wg2/ju26596/era5',
                f't2m_nhem_{year:04}-{month:02}.nc'
                )

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '2m_temperature',
                'year': f'{year:04}',
                'month': [f'{month:02}',],
                'day': [f'{day:02}' for day in range(1,days_in_month+1)],
                'time': ['00:00', '06:00', '12:00','18:00'],
                'grid': [1.0,1.0],
                'area': [
                    90, -180, 0,
                    180,
                ],
            },
            target_filename
            )
