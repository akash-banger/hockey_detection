from sklearn.cluster import KMeans
import easyocr
import cv2
from ultralytics import YOLO
import supervision as sv

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_number_dict = {}
        self.reader = easyocr.Reader(['en'])
        self.jersey_model = YOLO("models/jersey_model.pt")
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Add check for minimum number of players
        if len(player_colors) < 2:
            raise ValueError("Need at least 2 players to assign teams")
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id

    def get_player_jersey_number(self, frame, bbox, all_player_jersey_numbers):
        """
        Detect jersey number from player bbox using both posture detection and YOLO model.
        Returns None if no reliable detection.
        """
        # Convert jersey numbers to integers
        all_player_jersey_numbers = [int(jersey_number) for jersey_number in all_player_jersey_numbers]
        
        # Crop player image
        player_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Check player posture first
        gray = cv2.cvtColor(player_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour
        player_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(player_contour)
        
        # Calculate aspect ratio to determine posture
        aspect_ratio = float(w) / h
        
        # Only proceed with jersey detection if player is facing front/back
        if aspect_ratio >= 0.65:  # Not front/back facing
            return None
        
        # Use YOLO model for jersey number detection
        detections = self.jersey_model.predict(player_img)
        
        for detection in detections:
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Get detected digits and their x-coordinates
            digits = []
            for i in range(len(detection_supervision.xyxy)):
                x1 = detection_supervision.xyxy[i][0]  # x coordinate of left side
                digit = detection_supervision.data['class_name'][i]
                digits.append((x1, digit))
            
            # Sort digits by x-coordinate (left to right)
            digits.sort(key=lambda x: x[0])
            
            # Combine digits into number
            if digits:
                predicted_number = int(''.join(digit for _, digit in digits))
                
                # If predicted number is in the allowed list, return it
                if predicted_number in all_player_jersey_numbers:
                    return predicted_number
        
        return None