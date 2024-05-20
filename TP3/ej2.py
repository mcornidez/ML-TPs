import numpy as np
from PIL import Image
import cv2
from sklearn import svm

red = np.array(Image.open("Imgs/red.jpg"))
green = np.array(Image.open("Imgs/green.jpg"))
blue = np.array(Image.open("Imgs/blue.jpg"))
COLORS = np.array([red[0, 0], green[0, 0], blue[0, 0]])
print(COLORS)

def main():

    color_dim = 3
    sky = np.array(Image.open("Imgs/cielo.jpg"))
    sky = sky.reshape((sky.shape[0]*sky.shape[1], color_dim))
    grass = np.array(Image.open("Imgs/pasto.jpg"))
    grass = grass.reshape((grass.shape[0]*grass.shape[1], color_dim))
    cow = np.array(Image.open("Imgs/vaca.jpg"))
    cow = cow.reshape((cow.shape[0]*cow.shape[1], color_dim))

    #TODO hay que mezclar el dataset sin perder el orden de las labels, usando un mapa o algo asi
    perc = 0.8
    dataset = np.concatenate((sky, grass, cow), axis=0)
    labels = np.concatenate((0*np.ones(len(sky)), 1*np.ones(len(grass)), 2*np.ones(len(cow))))

    clf = svm.SVC()
    clf.fit(dataset[:int(perc*len(dataset))], labels[:int(perc*len(labels))])

    #Hacemos predicciones para ver que este dando buenos resultados
    #Falta tomar las metricas que pide la catedra y probar diferentes C y kernels
    predictions = clf.predict(dataset[-int((1-perc)*len(dataset)):])
    error = np.absolute(predictions-labels[-int((1-perc)*len(dataset)):]).sum()
    print(1 - error/len(predictions))

    #Hacemos predicciones sobre todo el dataset para pintar a la vaca lola
    #pic = np.array(Image.open("Imgs/horse.jpg"))
    #flat_pic = pic.reshape((pic.shape[0]*pic.shape[1], color_dim))
    #painting = clf.predict(flat_pic)
    #painting = np.array(list(map(lambda x: COLORS[int(x)], painting))).reshape((pic.shape[0], pic.shape[1], color_dim))
    #img = Image.fromarray(painting, 'RGB')
    #img.save('Imgs/horse_painting.jpg')



if __name__ == "__main__":
    main()
