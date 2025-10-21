import numpy as np

if __name__ == "__main__":
    dt = 1
    pointing_times = [893, 894, 895, 899]
    pointing_intervals = []

    interval = 0
    for j in range(1,len(pointing_times)):
        
        if pointing_times[j] - pointing_times[j-1] > dt:
            pointing_intervals.append(interval)
        else:
            interval += dt

    print(pointing_intervals)


