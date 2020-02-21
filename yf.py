from os import system
from time import time

def beep(start_time=None):
    duration = 1.4  # seconds
    freq = 400  # Hz

    if isinstance(start_time, float):
        print(f"Finished in {(time() - start_time)/60:.4} mins")

    system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
