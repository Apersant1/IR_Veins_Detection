
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

 
def seg(img, t=8, A=200,L=50):  

    # t: Порог => порог, используемый для сегментации изображения (значение 8-10 дает лучший результат. Значения Otsu и Isodata не приводят к лучшему результату)
    # A: Порог площади => все сегменты площадью меньше A удаляются и считаются шумом
    # L: Порог длины => все центральные линии длиной меньше L удаляются
    # Измените размер изображения на ~(1000px, 1000px) для лучшего результата
    

    # Выделение зеленого канала
    g = img[:,:,1]

    #Создание маски для ограничения поля зрения
    _, mask = cv2.threshold(g, 10, 255, cv2.THRESH_BINARY)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel, iterations=3)

    # CLAHE и очистка фона
    clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize=(9,9))
    g_cl = clahe.apply(g)
    g_cl1 = cv2.medianBlur(g_cl, 5)
    bg = cv2.GaussianBlur(g_cl1, (55, 55), 0)

    # Вычитание фона
    norm = np.float32(bg) - np.float32(g_cl1)
    norm = norm*(norm>0)
    
    # Пороговая сегментация
    _, t = cv2.threshold(norm, t, 255, cv2.THRESH_BINARY)

    # Удаление шума путем окрашивания контуров
    t = np.uint8(t)
    th = t.copy()
    contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if ( cv2.contourArea(c)< A):
            cv2.drawContours(th, [c], 0, 0, -1)
    th = th*(mask/255)
    th = np.uint8(th)
    #plt.imshow(th, cmap='gryay')  # THE SEGMENTED IMAGE

    # Преобразование расстояния для нахождения максимального диаметра
    vessels = th.copy()
    _,ves = cv2.threshold(vessels, 30, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(vessels, cv2.DIST_L2, 3)
    _,mv,_,mp = cv2.minMaxLoc(dist)
    print("Максимальный диаметр:",mv*2,"at the point:", mp)
    print("Выберите сосуд и нажмите Q после выбора.")

    # Извлечение центральной линии с использованием алгоритма тонкого сглаживания Zeun-Shang
    # Используется opencv-contrib-python, который предоставляет очень быстрый и эффективный алгоритм тонкого сглаживания
    # Пакет можно установить с помощью pip

    thinned = cv2.ximgproc.thinning(th)

    # Filling broken lines via morphological closing using a linear kernel
    kernel = np.ones((1, 10), np.uint8)
    d_im = cv2.dilate(thinned, kernel)
    e_im = cv2.erode(d_im, kernel) 
    num_rows, num_cols = thinned.shape
    for i in range (1, 360//15):
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 15*i, 1)
        img_rotation = cv2.warpAffine(thinned, rotation_matrix, (num_cols, num_rows))
        temp_d_im = cv2.dilate(img_rotation, kernel)
        temp_e_im = cv2.erode(temp_d_im, kernel) 
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -15*i, 1)
        im = cv2.warpAffine(temp_e_im, rotation_matrix, (num_cols, num_rows))
        e_im = np.maximum(im, e_im)

    # Skeletonizing again to remove unwanted noise
    thinned1 = cv2.ximgproc.thinning(e_im)
    thinned1 = thinned1*(mask/255)

    # Заполнение разорванных линий с помощью морфологического закрытия с использованием линейного ядра
    # Can be optimized further! (not the best implementation)
    thinned1 = np.uint8(thinned1)
    thh = thinned1
    hi = thinned1.copy()
    thi = thinned1.copy()
    hi = cv2.cvtColor(hi, cv2.COLOR_GRAY2BGR)
    thi = cv2.cvtColor(thi, cv2.COLOR_GRAY2BGR)
    thh = thh / 255

    kernel1 = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])
    kernel2 = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    kernel3 = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
    kernel4 = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    kernel5 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]

    for k in kernels:
        ks = [k] + [cv2.rotate(k, cv2.ROTATE_90_CLOCKWISE * i) for i in range(1, 4)]
        for kernel in ks:
            th = cv2.filter2D(thh, -1, kernel)
            indices = np.argwhere(th == 4.0)
            for i, j in indices:
                cv2.circle(hi, (j, i), 2, (0, 255, 0), 2)
                cv2.circle(thi, (j, i), 2, (0, 0, 0), 2)
    #plt.figure(figsize=(20, 14))
    thi = cv2.cvtColor(thi, cv2.COLOR_BGR2GRAY)
    #plt.imshow(hi, cmap='gray')  # This image shows all the bifurcation points

    # Удаление центральных линий, длина которых меньше L=50 px

    cl = thi.copy()
    contours, hierarchy = cv2.findContours(thi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if (c.size<L):
            cv2.drawContours(cl, [c], 0, 0, -1)


    # Центральная линия наложена на зеленый канал
    colors = [(100, 0, 150), (102, 0, 255), (0, 128, 255), (255, 255, 0), (10, 200, 10)]
    colbgr = [(193, 182, 255), (255, 0, 102), (255, 128, 0), (0, 255, 255), (10, 200, 10)] # цвет выделяемых линий
    
    im = g.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    thc = cl
    thh = thc.copy()
    thh = cv2.cvtColor(thh, cv2.COLOR_GRAY2BGR)
    contours, heirarchy = cv2.findContours(thc, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        
            color = np.random.randint(len(colors))
            cv2.drawContours(im, c, -1, colbgr[color], 2, cv2.LINE_AA)

            
            

  
    cv2.imwrite('result.png',cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    cv2.waitKey()
    cv2.destroyAllWindows()

    #Maximum diameter estimate
    d = mv*1.5
    
    return im, cl, d





