#%%
from flask import Flask, render_template, Response
import cv2


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


app = Flask(__name__)

#camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use 
camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)
            
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                #print(x,y,w,h)
                
                
                # Using cv2.putText() method 
                """image = cv2.putText(image, 'OpenCV', org, font,  
                                fontScale, color, thickness, cv2.LINE_AA) 
                """
                cv2.rectangle(
                    img=frame,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),
                    thickness=2
                )
            
            window_name = 'Image'
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (50, 50) 
            fontScale = 1
            color = (255, 0, 0) 
            thickness = 2    
            cv2.putText(img=frame, text="Vul test", 
                        org= org, 
                            fontFace= font,  
                            fontScale=fontScale, 
                            color=color, 
                            thickness=thickness, 
                            lineType=cv2.LINE_AA)  
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8010)
# %%
