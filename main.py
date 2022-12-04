import cv2
cascade_src_cars = 'cars.xml'
video_src = './VidSample\sample_01(Color).mp4'
cascade_src_two = 'two_wheeler.xml'
cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src_cars)
two_wheeler_cascade = cv2.CascadeClassifier(cascade_src_two)

while True:
    ret, img = cap.read()

    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    red = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    bikes = two_wheeler_cascade.detectMultiScale(gray, 1.1, 2)
    # for (x, y, w, h) in cars:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    for (x, y, w, h) in bikes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
