import time
import cv2
import numpy as np
# from pymycobot.mycobot import MyCobot
from tensorflow.keras.models import load_model
from pymycobot.mycobot import MyCobot
from inference_sdk import InferenceHTTPClient

# Inisialisasi MyCobot
bot = MyCobot('/dev/ttyTHS1', 1000000)
bot.power_on()
bot.init_eletric_gripper()

time.sleep(3)
bot.send_angles([0,0,0,0,0,0],40)

# time.sleep(3)
# bot.set_gripper_value(100,30,1)

# time.sleep(3)
# bot.send_angles([0,0,-65,0,0,135],40)

time.sleep(5)
webcam = cv2.VideoCapture('/dev/video0')
if webcam is None :
    print("Gagal membuka webcam")
    exit()

model = load_model('fruit_classification_model.h5')
class_names = ['Apple', 'avocado', 'banana', 'chery fruit', 'grape', 'mango fruit', 'orange', 'ressberry']

# Inisialisasi MyCobot
bot = MyCobot('/dev/ttyTHS1', 1000000)
bot.power_on()
bot.init_eletric_gripper()

time.sleep(3)
bot.send_angles([0,0,0,0,0,45],40)

time.sleep(3)
bot.set_gripper_value(100,30,1)

time.sleep(3)
bot.send_angles([0,0,-65,0,0,45],40)

print('Inisiasi Robot')
time.sleep(5)

Bananacount = 0
Garapecount = 0


while True:
    ret, imageFrame = webcam.read()
    if not ret:
        print("Gagal menangkap gambar")
        break

    resized_frame = cv2.resize(imageFrame, (100, 100))
    processed_frame = resized_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)

    predictions = model.predict(processed_frame)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = predictions[0][class_index]

    if class_name == 'banana':
        Bananacount += 1
    
    if class_name == 'grape':
        Garapecount += 1

    if Garapecount > 10 :
        time.sleep(3)
        bot.send_angles([0,-83,0,0,0,45],40)

        time.sleep(3)
        bot.set_gripper_value(0,30,1)

        time.sleep(3)
        bot.send_angles([-90,-45,0,0,0,45],40)

        time.sleep(3)
        bot.set_gripper_value(100,30,1)

        time.sleep(5)
        print('Robot menaruh Anggur ke tempatnya')
        time.sleep(5)
        Garapecount = 0
        Bananacount = 0
        time.sleep(3)
        bot.send_angles([0,0,0,0,0,45],40)

        time.sleep(3)
        bot.set_gripper_value(0,30,1)

    if Bananacount> 6:
        time.sleep(3)
        bot.send_angles([0,-83,0,0,0,45],40)

        time.sleep(3)
        bot.set_gripper_value(0,30,1)

        time.sleep(3)
        bot.send_angles([90,-45,0,0,0,45],40)

        time.sleep(3)
        bot.set_gripper_value(100,30,1)
        time.sleep(5)
        print('Robot menaruh Pisang ke tempatnya')
        time.sleep(5)
        Bananacount = 0
        Garapecount = 0
        time.sleep(3)
        bot.send_angles([0,0,0,0,0,45],40)

        time.sleep(3)
        bot.set_gripper_value(100,30,1)
        webcam.release()
        time.sleep(3)
        bot.send_angles([0,0,-65,0,0,45],40)
        time.sleep(5)
        webcam = cv2.VideoCapture('/dev/video0')
    # Menambahkan teks ke gambar
    text = f'Class: {class_name}, Confidence: {confidence:.2f}'
    cv2.putText(imageFrame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Fruit Detection in Real-Time", imageFrame)

    print(f'Class: {class_name}, Confidence: {confidence:.2f}')
    print(f'Banana Count: {Bananacount}')
    print(f'Grape Count: {Garapecount}')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    if webcam is None or not webcam.isOpened():
        print("gagal")

cv2.destroyAllWindows()
