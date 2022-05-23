import os

folder = r'/home/bouchra/Project1/unet-data/output2/seg3/'
count = 0
for file_name in os.listdir(folder):
    source = folder + file_name
    destination = folder + str(count) + ".tif"
    os.rename(source, destination)
    count += 1

print('All Files Renamed')