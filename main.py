import cv2
import pathlib

cascade_human = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
cascade_cat = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalcatface.xml"

human = cv2.CascadeClassifier(str(cascade_human))
cat = cv2.CascadeClassifier(str(cascade_cat))

camera = cv2.VideoCapture(1)
camera.set(3,600)
camera.set(4,600)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hface = human.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    cface = cat.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for (x,y,width,height) in hface:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0), 2)
        cv2.putText(frame,'Human',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)

    for (x,y,width,height) in cface:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0), 2)
        cv2.putText(frame,'Cat',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
