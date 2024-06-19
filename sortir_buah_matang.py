from inference_sdk import InferenceHTTPClient
import time
import cv2
from pymycobot.mycobot import MyCobot
from inference_sdk import InferenceHTTPClient

# Inisialisasi MyCobot
bot = MyCobot('/dev/ttyTHS1', 1000000)
bot.power_on()
bot.init_eletric_gripper()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="QNMHdJp4flhgePTgtTvu"
)

time.sleep(3)
bot.send_angles([0,0,0,0,0,45],40)

time.sleep(3)
bot.set_gripper_value(100,30,1)

time.sleep(3)
bot.send_angles([0,0,-65,0,0,45],40)

time.sleep(5)
webcam = cv2.VideoCapture('/dev/video0')
if webcam is None or not webcam.isOpened():
    print("gagal")

while True:
    
    __, imageFrame = webcam.read()
    cv2.imshow("Fruit Detection in Real-TIme", imageFrame) 
    result = CLIENT.infer(imageFrame , model_id="fruitclassification-jx6nu/1")
    if result['predictions'] == []:
        print('no fr')

    elif result['predictions'] != [] :
        predicted_class = result['predictions'][0]['class']
        print(predicted_class)
        webcam.release()
        
        if predicted_class == 'banana_good':
            time.sleep(3)
            bot.send_angles([0,-83,0,0,0,45],40)

            time.sleep(3)
            bot.set_gripper_value(0,30,1)

            time.sleep(3)
            bot.send_angles([90,-45,0,0,0,45],40)

            time.sleep(3)
            bot.set_gripper_value(100,30,1)

        if predicted_class == 'guava_good':
            time.sleep(3)
            bot.send_angles([0,-83,0,0,0,45],40)

            time.sleep(3)
            bot.set_gripper_value(0,30,1)

            time.sleep(3)
            bot.send_angles([-90,-45,0,0,0,45],40)

            time.sleep(3)
            bot.set_gripper_value(100,30,1)

        time.sleep(3)
        bot.send_angles([0,0,-65,0,0,45],40)
        time.sleep(5)
        webcam = cv2.VideoCapture('/dev/video0')
        if webcam is None or not webcam.isOpened():
            print("gagal")

    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break



time.sleep(3)
bot.set_gripper_value(100,30,3)

time.sleep(3)
bot.send_angles([0,0,0,0,0,45],40)
print(bot.get_coords())


cv2.destroyAllWindows()