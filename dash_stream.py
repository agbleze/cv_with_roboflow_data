import dash, cv2
import dash_core_components as dcc
#import dash_html_components as html
#import mediapipe as mp
from flask import Flask, Response
from dash import html

#mp_drawing = mp.solutions.drawing_utils
#mp_pose = mp.solutions.pose

class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

"""    def get_frame(self):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            success, image = self.video.read()

            # Recolor image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )

            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
"""

def gen(camera):
    while True:
        video_capture = cv2.VideoCapture(camera)
        ret, frame = video_capture.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        #frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed_1')
def video_feed_1():
    return Response(gen(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/video_feed_2')
def video_feed_2():
    return Response(gen(0), mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.Img(src="/video_feed_1", style={'width' : '40%', 'padding': 10}),
   # html.Br(),
    #html.Img(src="/video_feed_2", style={'width' : '40%', 'padding': 10})
])

if __name__ == '__main__':
    app.run_server(debug=True)