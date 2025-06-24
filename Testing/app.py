from flask import Flask, render_template
import xarray as xr
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    nc_path = 'data/ch.nc'

    if not os.path.exists(nc_path):
        return "NetCDF file not found!"

    # Load dataset
    ds = xr.open_dataset(nc_path)

    # Plot if variable exists
    if 't2m' in ds:
        t2m = ds['t2m'].mean(dim=['longitude', 'latitude'])

        plt.figure(figsize=(10, 5))
        t2m.plot(label='2m Temperature')
        plt.title('Forecasted 2m Temperature (Ensemble Mean)')
        plt.ylabel('Temperature (K)')
        plt.xlabel('Time')
        plt.grid()
        plt.legend()

        # Save the plot
        plot_path = 'static/forecast.png'
        plt.savefig(plot_path)
        plt.close()
    else:
        return "Variable 't2m' not found in the dataset."

    return render_template('index.html', image_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
