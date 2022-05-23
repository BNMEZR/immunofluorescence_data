import os

folder = r'/path/to/file/'
count = 0
for file_name in os.listdir(folder):
    source = folder + file_name
    destination = folder + str(count) + ".tif"
    os.rename(source, destination)
    count += 1

print('All Files Renamed')
