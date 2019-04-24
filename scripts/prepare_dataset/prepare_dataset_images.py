import os

os.chdir('./dataset')
imdir = 'images'

if not os.path.isdir(imdir):
    os.mkdir(imdir)
    
dataset = [folder for folder in os.listdir('.') if 'images' not in folder]

print(dataset)

n = 0
for folder in dataset:
    for imfile in os.scandir(folder):
        os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(n)))
        n += 1

    
    