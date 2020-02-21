import platform
import os
from time import time


def beep(start_time=None):
    sys = platform.system()
    duration = 1.6  # seconds
    freq = 380      # Hz
    
    if sys == 'Linux' or sys == 'Darwin':
        try:
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        except:
            print('Please install sox to enable beep sound!')
    elif sys == 'Windows':
        import winsound
        winsound.Beep(freq, duration*1000)
    else:
        print('\007')

    if isinstance(start_time, float):
        print(f"Finished in {(time() - start_time)/60:.4} mins")

