import os
file_list = os.listdir('/root/autodl-tmp/assets/phototourism_sample_images')
f = open('name.txt', 'w')
file_list.sort()
for i in range(0, len(file_list), 2): 
    f.write(file_list[i] + " " + file_list[i + 1] + '\n')
f.close()