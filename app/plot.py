import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
def do_plot(vec):
    f = plt.plot(vec)
    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image