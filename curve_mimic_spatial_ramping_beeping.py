from ultralytics import YOLO
import cv2
import math 
import torch
import time
import numpy as np
import logging
import sys
from queue import Queue, Empty
import subprocess
import threading
import sounddevice as sd
import wave
import tempfile
import os
import glob
import multiprocessing
from scipy.optimize import minimize
from scipy.io.wavfile import read, write
from scipy.io import wavfile
# start webcam#
video_source = 0
#video_source = "curb_video1.mp4"
orientation_mimic = False

def generate_camera_output_filename(base_dir=".", prefix="camera", extension=".mp4"):
    # Find all existing files that match the pattern
    existing_files = glob.glob(os.path.join(base_dir, f"{prefix}_*{extension}"))
    # Extract indices from filenames
    indices = [int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) for file in existing_files if file.split('_')[-1].replace(extension, '').isdigit()]
   
    # Determine the next index
    next_index = max(indices) + 1 if indices else 1
   
    # Generate the new filename
    new_filename = os.path.join(base_dir, f"{prefix}_{next_index}{extension}")
    return new_filename
    
if isinstance(video_source, str):
    name, extension = os.path.splitext(video_source)
    output_file=f"{video_source}_result{extension}"
elif isinstance(video_source, int):
    output_file = generate_camera_output_filename()
     
#cap = cv2.VideoCapture(video_source)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

data = np.load('calibration_parameters.npz')
mtx = data['camera_matrix']
dist = data['distortion_coefficients']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device:{device}")
# model
model = YOLO("seg_best_4.pt").to(device)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(frame_width,frame_height),0,(frame_width,frame_height))
print(frame_width, frame_height)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / frame_rate) if frame_rate > 0 else 25

#setup the output video writer

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, frame_rate,(frame_width,frame_height))
# object classes
classNames = ["curb"]
color_mapping = {
    range(0, 3): ((0, 0, 255), 1),  # Red
    range(3, 6): ((0, 225, 255), 2),  # Yellow
    range(6, 9): ((0, 225, 0), 3)  # Green
}

position_direction_mapping = {
    0: 'left',
    1: 'front',
    2: 'right',
    3: 'left',
    4: 'front',
    5: 'right',
    6: 'left',
    7: 'front',
    8: 'right'
}

def get_color_and_object_color(fan_area):
    for key, value in color_mapping.items():
        if fan_area in key:
            return value
    return (255, 0, 255), None  # Default to Purple if not found

def get_position_direction(fan_area):
    return position_direction_mapping.get(fan_area, "")
    
class BeepThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.distance = None
        self.running = True
        self.update_interval = 0.01
        self.object_color = None
        self.processes = []
        self.sound_config = {
            1: {'frequency': 500, 'volume': 1.2, 'duration': 0.05, 'multiplier': 0.683, 'distance_factor': 410, 'bias': -0.284, 'reverberation': 20},
            2: {'frequency': 300, 'volume': 1, 'duration': 0.06, 'multiplier': 1.8375, 'distance_factor': 490, 'bias': -1.0375, 'reverberation': 30},
            3: {'frequency': 200, 'volume': 0.8, 'duration': 0.07, 'multiplier': 3.867, 'distance_factor': 580, 'bias': -2.37, 'reverberation': 40}
        }
        self.beep_timestamps = []  # List to store timestamps of each beep
        self.last_audio_play_time = 0
        self.min_x_value = None
        self.beeping_pan_x = 0.5
    
    def cleanup_process(self):
        self.processes = [p for p in self.process if p.poll() is None]
        
    def beep(self, frequency, volume, duration, multiplier, distance_factor, bias, reverberation, beeping_pan_x):
        try:
            
            if self.distance is not None and self.object_color in self.sound_config:

                total_delay = self.distance * multiplier / distance_factor + bias
                #total_delay = 
                if frequency == 200:
                    if total_delay < 1.05:
                        total_delay = 0.9
                        frequency = 250
                    elif  total_delay <1.2:
                        total_delay = 1.1
                        frequency = 235
                    elif total_delay < 1.35:
                        total_delay = 1.3
                        frequency = 220
                    else: 
                        total_delay = 1.5
                        frequency = 205          
                
                elif frequency == 300:
                    if  total_delay < 0.6:
                        total_delay = 0.5
                        frequency = 400
                    
                    elif total_delay < 0.7:
                        total_delay = 0.65
                        frequency = 350
                    else: 
                        total_delay = 0.8
                        frequency = 300
                #reverb_params = "reverb 50 50 100"
                #current_time = time.time()  # Get current time

                right_channel = beeping_pan_x # Equal power pan law for left channel
                left_channel= 1-beeping_pan_x  # Equal power pan law for right channe
                print(right_channel,left_channel)
                #self.beep_timestamps.append(current_time)  # Record timestamp
                process = subprocess.Popen(["play", "-n", "vol", str(volume), "synth", str(duration), "sin", str(frequency), "gain", "-2", "reverb", str(reverberation),"remix", "-m","1v{}".format(left_channel),"2v{}".format(right_channel)]) 
                self.processes.append(process)
                elapsed_delay = 0
                while elapsed_delay < total_delay- 0.1 and self.running:
                    time.sleep(self.update_interval)
                    elapsed_delay += self.update_interval
                    #print(elapsed_delay)
        except Exception as e:
            print(f"Error in beep method: {e}")

    def run(self):
        print("running beeping")
        while self.running:
            if self.object_color in self.sound_config and self.distance is not None:
                config = self.sound_config[self.object_color]
                
                print("beeping____________________________________________")
                self.beep(config['frequency'], config['volume'], config['duration'], config['multiplier'], config['distance_factor'], config['bias'], config['reverberation'],beeping_pan_x)
                #self.cleanup_processes()
            else:
                self.distance = None
                self.object_color = None
                #print("no beep______________________________________________________________")
                time.sleep(0.01)
            time.sleep(self.update_interval)  # Adjust based on your needs


    def update_distance(self, new_distance, new_color=None, beeping_pan_x=None):
        self.distance = new_distance
        self.object_color = new_color
        self.beeping_pan_x = beeping_pan_x
        

    def stop(self):
        print("beeping stopping working!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
        for process in self.processes:
            process.terminate()
        self.running = False

    def calculate_intervals(self):
        intervals = []
        for i in range(1, len(self.beep_timestamps)):
            interval = self.beep_timestamps[i] - self.beep_timestamps[i-1]
            intervals.append(interval)
        return intervals
    
                
class AudioPlaybackThread(threading.Thread):
    def __init__(self, audio_folder_path):
        super().__init__()
        self.audio_folder_path = audio_folder_path
        self.playback_queue = Queue()
        self.stop_request = threading.Event()
        self.last_audio_play_time = time.time() - 3  # Initialize to allow immediate playback

    def create_stereo_pan(self, audio, start_point, end_point, frame_width):
        
        start_pan = (start_point / frame_width) * 2 - 1
        end_pan = (end_point / frame_width) * 2 - 1
        pan_curve = np.linspace(start_pan, end_pan, len(audio))
        stereo_signal = np.zeros((len(audio), 2))
        for i, pan in enumerate(pan_curve):
            left_gain = (1 - pan) ** 2
            right_gain = (1 + pan) ** 2
            norm_factor = np.sqrt(left_gain + right_gain)
            left_gain /= norm_factor
            right_gain /= norm_factor
            stereo_signal[i, 0] = audio[i] * left_gain
            stereo_signal[i, 1] = audio[i] * right_gain
        max_amplitude = np.max(np.abs(stereo_signal))
        if max_amplitude > 0:
            stereo_signal /= max_amplitude
        stereo_signal = np.int16(stereo_signal * 32767)
        return stereo_signal

    def run(self):
        while not self.stop_request.is_set():
            try:
                request = self.playback_queue.get(timeout=0.1)  # Adjust timeout as needed
                if request is None:  # Check if the request is the sentinel value for stopping the thread
                    break  # Exit the loop
                audio_file_name, start_point, end_point, frame_width = request
                audio_file_path = f"{self.audio_folder_path}/{audio_file_name}"
                fs, mono_audio = read(audio_file_path)
                stereo_audio = self.create_stereo_pan(mono_audio, start_point, end_point, frame_width)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                write(temp_file.name, fs, stereo_audio)
                subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_file.name])
            except Empty:
                continue

    def enqueue_audio_playback(self, audio_file_name, start_point, end_point, frame_width):
        current_time = time.time()
        if current_time - self.last_audio_play_time >= 3:
            self.playback_queue.put((audio_file_name, start_point, end_point, frame_width))
            self.last_audio_play_time = current_time
    def stop(self):
        self.stop_request.set()
        self.playback_queue.put(None)  # Ensure the thread exits from the blocking get call
        self.join()
                                   
class TextToSpeech(threading.Thread):#mimic3
    def __init__(self, mimic3_path, logger,voice='default' ,delay=4):
        threading.Thread.__init__(self)
        self.mimic3_path = mimic3_path
        self.logger = logger
        self.voice = voice
        self.delay = delay
        self.phrases = Queue()
        self.exit_flag = False
        self.process = None
        self.speaking_done_event = threading.Event()
        self.last_spoken_phrase = None
        self.last_spoken_time = 0
        self.repeat_delay = delay
        self.ready_event = threading.Event() 
    
    def run(self):
        self.process = subprocess.Popen([self.mimic3_path, '--interactive','--voice', self.voice],
                                        stdin=subprocess.PIPE,
                                        text=True)
        self.ready_event.set()

        while not self.exit_flag:
            if not self.phrases.empty():
                phrase = self.phrases.get()
                self.speak(phrase)
                print("speak")
                self.speaking_done_event.clear()
            
            else:
                self.speaking_done_event.set()
            time.sleep(self.delay)
        
        self.process.terminate()

    def speak(self, phrase):
        try:
            self.process.stdin.write(phrase + '\n')
            self.process.stdin.flush()       
            
        except Exception as e:
            self.logger.error("Error in speaking: " + str(e))

    def enqueue_phrase(self, phrase):
        #with self.phrases.mutex:
        self.phrases.queue.clear()
        self.phrases.put(phrase)
  
    def stop(self):
        self.exit_flag = True
    
    def wait_until_speaking_done(self):
        self.speaking_done_event.wait() 
    
    def clear_queue(self):
        #with self.phrases.mutex:
        self.phrases.queue.clear()

import math

class FanZoneDetector: #decide which area the object locates in
    def __init__(self, center, red_axes, yellow_axes, green_axes, start_angle, end_angle, mid_left_angle, mid_right_angle, device):
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.axes_lengths = [red_axes, yellow_axes, green_axes]
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.mid_left_angle = mid_left_angle
        self.mid_right_angle = mid_right_angle
        self.device = device
        
    def calculate_line(self, x_radius,y_radius, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        x = self.center[0] + x_radius * math.cos(angle_radians)
        y = self.center[1] + y_radius * math.sin(angle_radians)  
        return int(x), int(y)

    def classify_points(self,points):
        """
        Classify points based on their location relative to elliptical fan areas and angular boundaries.
        """
        # Convert points to a PyTorch tensor and move to the specified device
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        # Calculate the displacement of each point from the center
        displacement = points_tensor - self.center
        # Calculate the distance of each point from the center
        distances = torch.norm(displacement, dim=1)
        #print(distances)
        # Calculate the angle for each point
        angles = torch.atan2(displacement[:, 1], displacement[:, 0]) * (180 / np.pi)
        angles = (angles + 360) % 360 -360  # Ensure angles are within [0, 360)
        #print(angles)
        # Initialize classification tensor
        classifications = torch.full((points_tensor.shape[0],), -1, dtype=torch.int, device=device)
        closest_distance = float('inf')
        closest_zone = None
        # Process each fan area
        for i, (a, b) in enumerate(reversed(self.axes_lengths)):
        # Check if points are within the ellipse defined by a and b
            within_ellipse = ((displacement[:, 0] ** 2) / (a ** 2) + (displacement[:, 1] ** 2) / (b ** 2)) <= 1
            
        # Check if points are within the angular boundaries
            within_angle = (angles >= self.start_angle) & (angles <= self.end_angle)
            if i==0:
            
                valid_points = within_ellipse & within_angle
        # Further classify within the fan zone based on mid angles
            mid_left = (angles >= self.start_angle) & (angles < self.mid_left_angle)
            mid_right = (angles > self.mid_right_angle) & (angles <= self.end_angle)
            mid = ((angles < self.mid_right_angle) & (angles > self.mid_left_angle))
            # Update classifications
            classifications[within_ellipse & mid_left] = 6-3*i
            classifications[within_ellipse & mid] = 7-3*i
            classifications[within_ellipse & mid_right] = 8-3*i
        #print(classifications)
        if valid_points.any():
            valid_distances = distances[valid_points]
            min_distance, min_idx_relative = torch.min(valid_distances, 0)
    
            # Convert the boolean mask to indices
            valid_indices = torch.nonzero(valid_points, as_tuple=True)[0]
    
            # Use the relative index to get the actual index in the original dataset
            min_idx = valid_indices[min_idx_relative]
            min_x_value = points[min_idx, 0].item()
            # Now, min_idx corresponds to the original dataset
            closest_zone_classification = classifications[min_idx]
            
            
            #print(closest_zone_classification)
        else:
            return None, None, None, None
        return classifications.cpu().numpy(), min_distance, closest_zone_classification, min_x_value
    
# Fan properties
fan_center_x = frame_width // 2
#fan_center_y = frame_height + 100  # Adjust the Y value as needed to position the center below the frame
fan_center_y = frame_height + 290
fan_center = (fan_center_x, fan_center_y)
if frame_height == 1080:
    print("1080 cone")
    # Parameters for 1080p resolution
    start_circle = 500
    circle_distance = 150  # Corrected variable name from 'circle_distane' to 'circle_distance'
    green_line_radius = 800
    # Define axes lengths for 1080p resolution
    axes_lengths = [
        (start_circle - 150, start_circle + 10),  # Smallest fan (red)
        (start_circle + circle_distance - 130, start_circle + circle_distance),  # Medium fan (Yellow)
        (green_line_radius - 30, green_line_radius)  # Largest fan (green)
    ]
    start_angles = (-117,-114,-110)
    end_angles = (-63,-66,-70)
else:
    print("480 cone")
    # Parameters for 480p resolution
    start_circle = 400  # It seems you've defined 'start_circle' twice with different values for 480p. You might want to use one.
    # start_circle = 400  # This line seems to be a mistake as 'start_circle' is redefined. Comment or remove if not needed.
    circle_distance = 90
    green_line_radius = 580 
    # Define axes lengths for 480p resolution
    axes_lengths = [
        (start_circle - 150, start_circle + 10),  # Smallest fan (red)
        (start_circle + circle_distance - 130, start_circle + circle_distance),  # Medium fan (Yellow)
        (green_line_radius - 30, green_line_radius)  # Largest fan (green)
    ]
    start_angles = (-120,-115,-110) #for 640*480
    end_angles = (-60,-65,-70)
# Fan colors
fan_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # Green, Yellow, Red
#Fab angles


# Angles for the fans
#start_angle = -120  # Starting angle
#end_angle = -60     # Ending angle
start_angle = -110
end_angle = -70

mid_left_angle = -100
mid_right_angle = -80

fan_detector = FanZoneDetector(fan_center, axes_lengths[0], axes_lengths[1], axes_lengths[2], start_angle, end_angle, mid_left_angle, mid_right_angle, device=device)
left_line = fan_detector.calculate_line(axes_lengths[-1][0],axes_lengths[-1][1], start_angle)
right_line = fan_detector.calculate_line(axes_lengths[-1][0],axes_lengths[-1][1], end_angle)
mid_left_line = fan_detector.calculate_line(axes_lengths[-1][0],axes_lengths[-1][1], mid_left_angle)
mid_right_line = fan_detector.calculate_line(axes_lengths[-1][0],axes_lengths[-1][1], mid_right_angle)
prev_frame_time = 0
new_frame_time = 0


log_format = "%(levelname)s %(name)s - %(message)s"
formatter = logging.Formatter(log_format)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
log_filter = logging.Filter("root")
stream_handler.addFilter(log_filter)
logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

pos_and_direction = None
last_spoken_area = None
object_color = None

beep_thread = BeepThread()
beep_thread.daemon = True
beep_thread.start()

audio_folder_path = "Audio/vertical"
audio_playback_thread = AudioPlaybackThread(audio_folder_path)
audio_playback_thread.start()

mimic3_tts_thread = TextToSpeech(mimic3_path="mimic3", logger=logger, voice= "en_US/cmu-arctic_low#aew")
mimic3_tts_thread.daemon = True
mimic3_tts_thread.start()
mimic3_tts_thread.ready_event.wait()

sample_rate = 44100
total_duration = 1.0 
#time.sleep(3)

mimic3_tts_thread.speak("starting")
degree = 3
confidence_threshold = 0.05

while True:
    success, img = cap.read()
    if not success:
        break   
    new_frame_time = time.time()

    dst=cv2.undistort(img,mtx,dist,None,newcameramtx)

    #dst = cv2.flip(dst,-1)
    confidence_i = None
    lowest_mask = None
    results = model(img, stream=True)
    for result in results:
        if result.masks is not None:
            # Convert bounding boxes and masks to numpy arrays if they're not already
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            # Initialize variables to track the lowest box and its mask
            lowest_y2 = -1
            lowest_mask_index = None

            # Iterate through the boxes to find the one that extends the lowest
            for i, box in enumerate(boxes):
                if confidences[i] >= confidence_threshold:
                	x1, y1, x2, y2 = box
                	if y2 > lowest_y2:
                    		lowest_y2 = y2
                    		lowest_mask_index = i
                    		confidence_i = confidences[i]
            
            # Use the index of the lowest box to find the corresponding mask
            if lowest_mask_index is not None:
                lowest_mask = masks[lowest_mask_index]
                #print(lowest_mask)          
                curb_mask_np = (lowest_mask * 255).astype(np.uint8)
                cv2.putText(dst, f"confidence:{confidence_i:.2f}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)


    # make the feedback
    if lowest_mask is not None:
        curb_mask_resized = cv2.resize(curb_mask_np, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        edges = cv2.Canny(curb_mask_resized, 100, 200)
        edge_points = np.where(edges != 0)
        #print(edge_points)
        if edge_points[0].size > 0:  # Check if there are any edges
        # Fit a polynomial curve
            
            coefficients = np.polyfit(edge_points[1], edge_points[0], degree)
            polynomial = np.poly1d(coefficients) 
            derivative = np.polyder(polynomial)
        # Generate x and y values for the curve
            x_vals = np.linspace(min(edge_points[1]), max(edge_points[1]), 1000)
            y_vals = polynomial(x_vals)
            points = np.column_stack((x_vals, y_vals))
            gradients = derivative(x_vals)
            average_gradient = np.mean(gradients)
            angle_radians = np.arctan(average_gradient)
            angle_degrees = -int(np.degrees(angle_radians))
            start_point = x_vals[0]
            end_point = x_vals[-1]
            
            if angle_degrees >0:
            	rounded_angle = (angle_degrees // 5)*5
            else:
                positive_angle = abs(angle_degrees)
                rounded_angle = -(positive_angle // 5) * 5           
            
            print(f"Average gradient: {angle_degrees}")
            print("Corresponding angle in degrees:", rounded_angle)
        # Draw the curve on the image
            for i in range(1, len(x_vals)):
                cv2.line(dst, (int(x_vals[i-1]), int(y_vals[i-1])), (int(x_vals[i]), int(y_vals[i])), (0, 0, 255), 2)
            
            classifications, min_distance, fan_area, min_x_value = fan_detector.classify_points(points)
            
            #if classifications is not None:
                #change_indices = np.where(np.diff(classifications) != 0)[0] + 1
                #unique_sequence = np.concatenate(([classifications[0]], classifications[change_indices]))
                
                
        if fan_area is not None:
            print(fan_area)
            beeping_pan_x = min_x_value/frame_width
            color, object_color = get_color_and_object_color(fan_area)
            beep_thread.update_distance(min_distance, object_color, beeping_pan_x)

            if orientation_mimic is True:
                # Determine the direction based on the angle
                if rounded_angle == 0:
                    orientation_text = "horizontal"
                elif rounded_angle == 90:
                    orientation_text = "vertical"
                else:
                    direction = "left" if rounded_angle > 0 else "right"
                    adjusted_angle = abs(rounded_angle)
                    orientation_text = f"{adjusted_angle} {direction}"
                mimic3_tts_thread.enqueue_phrase(orientation_text)
                print(orientation_text)
            
            else:
                pos_and_direction = get_position_direction(fan_area)
                mimic3_tts_thread.enqueue_phrase(pos_and_direction)
                if color != (255, 0, 255):
                    audio_playback_thread.enqueue_audio_playback(f"{rounded_angle}.wav", start_point, end_point, frame_width) 
        else:
            color = (255, 0, 255)  # Purple
            beep_thread.update_distance(None,None,None)  
        curve_points = np.array([x_vals, y_vals]).T.round().astype(int).reshape(-1, 1, 2)    
        cv2.polylines(dst, [curve_points], False, color, 2)
        color_overlay = np.array(color)
            # Apply the mask to the undistorted image
        mask_bool = lowest_mask.astype(bool)
        mask_bool_resized = cv2.resize(mask_bool.astype(np.uint8), (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool_resized = mask_bool_resized.astype(bool)
        dst[mask_bool_resized] = dst[mask_bool_resized] * 0.7 + color_overlay * 0.3
            # Prepare the color overlay
            # This creates a (3,) array, e.g., for blue: [255, 0, 0]

        
            # Apply the color overlay where the mask is True
        #dst[mask_bool] = dst[mask_bool] * 0.7 + color_overlay * 0.3
            #cv2.imshow('Lowest Mask Overlay', dst)
            #if cv2.waitKey(1) & 0xFF == ord('w'):
            #    break
        
                  
    else:
        print("No mask found.")
        beep_thread.update_distance(None,None,None)
        
        #mimic3_tts_thread.enqueue_phrase("no object")        
    
    #Visualization
        # Draw the concentric fans
    for axes, color, start_angle, end_angle in zip(axes_lengths, fan_colors, start_angles, end_angles):
        cv2.ellipse(dst, fan_center, axes, 0, start_angle, end_angle, color, 1)
   
    # White color for the bordrey line
    cv2.line(dst, fan_center, left_line, (255, 255, 255), 1)  
    cv2.line(dst, fan_center, right_line, (255, 255, 255), 1)
    
    cv2.line(dst, fan_center, mid_left_line, (255, 255, 255), 1)
    cv2.line(dst, fan_center, mid_right_line, (255, 255, 255), 1)
         
    fps = int(1/(new_frame_time - prev_frame_time))
    prev_frame_time= new_frame_time
    fps_text = f"FPS: {fps}"
    cv2.putText(dst, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        
    out.write(dst)
    cv2.namedWindow('world', cv2.WINDOW_NORMAL)
    cv2.imshow('world', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#intervals = beep_thread.calculate_intervals()
beep_thread.stop()
beep_thread.join()
#print("Intervals between beeps:", intervals)
mimic3_tts_thread.stop()
mimic3_tts_thread.join()
audio_playback_thread.stop()
audio_playback_thread.join()
cap.release()
out.release()
cv2.destroyAllWindows()
