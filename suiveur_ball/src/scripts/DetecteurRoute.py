#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from ultralytics import YOLO

class DetecteurRoute:
    def __init__(self):
        rospy.init_node('detecteur_route_node', anonymous=False)

        # Classes YOLO
        self.CLASS_DOUBLE = 0
        self.CLASS_JAUNE = 4
        self.CLASS_PARKING = 5
        self.CLASS_PIETON = 6
        self.CLASS_ROBOT = 7
        self.CLASS_ROUTE = 8 

        # Chargement modele
        path_model = "/home/ubuntu/catkin_ws/src/suiveur_ball/src/scripts/best_seg.onnx"
        self.model = YOLO(path_model, task='segment')

        # Paramètres
        self.imsize = 128
        self.SCAN_CONDUITE = 105   
        self.etape_mission = 0  
        self.derniere_erreur = 0
        self.cible_parking = 64 
        self.dernier_temps = time.time()
        self.prochaine_manoeuvre_autorisee = time.time() + 1.0 
        self.timer_arret = 0
        self.temps_traversée = 0

        self.sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/road_error', Point, queue_size=1)

    def get_center_of_biggest_blob(self, mask, y_start, height=15):
        roi = mask[y_start : y_start + height, :]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 5: return None
        M = cv2.moments(largest_contour)
        return int(M["m10"] / M["m00"]) if M["m00"] > 0 else None

    def callback(self, msg):
        try:
            now = time.time()
            dt = now - self.dernier_temps
            self.dernier_temps = now

            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.imsize, self.imsize))
            results = self.model.predict(img, imgsz=self.imsize, conf=0.25, verbose=False, device='cpu')
            
            masks = {k: np.zeros((self.imsize, self.imsize), dtype=np.uint8) for k in [0,4,5,6,7,8]}
            if results[0].masks is not None:
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls.cpu())
                    if cls in masks:
                        pts = results[0].masks[i].xy[0].astype(int)
                        if len(pts) > 0: cv2.fillPoly(masks[cls], [pts], 255)

            stop_code = 1.0 
            err_finale = self.derniere_erreur
            facteur_vitesse = 1.0 

            # 1. VITESSE ADAPTATIVE si robot devant
            area_robot = np.count_nonzero(masks[self.CLASS_ROBOT])
            if area_robot > 2800: facteur_vitesse = 0.0
            elif area_robot > 1500: facteur_vitesse = max(0.3, 1.0 - ((area_robot - 1500) / 1300))

            # 2. NAVIGATION (Ligne Jaune ou Route)
            cx_j = self.get_center_of_biggest_blob(masks[self.CLASS_JAUNE], self.SCAN_CONDUITE)
            
            if self.etape_mission == 3:
                self.temps_traversée += dt
                if self.temps_traversée < 2.5: 
                    err_finale = 0 # Tout droit
                elif self.temps_traversée < 6.0: 
                    cx_r = self.get_center_of_biggest_blob(masks[self.CLASS_ROUTE], self.SCAN_CONDUITE)
                    err_finale = (cx_r - 64) if cx_r is not None else 0
                else:
                    self.etape_mission = 0
                    self.prochaine_manoeuvre_autorisee = now + 1.0
            else:
                if cx_j is not None:
                    err_finale = (cx_j + 45) - 64
            
            self.derniere_erreur = err_finale

            # 3. DETECTION PARKING
            if self.etape_mission == 0 or (self.etape_mission == 3 and self.temps_traversée > 3.0):
                area_p = np.count_nonzero(masks[self.CLASS_PARKING])
                if area_p > 250:
                    M = cv2.moments(masks[self.CLASS_PARKING])
                    if M["m00"] > 0:
                        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        self.cible_parking = cX 
                        if cY < 105:
                            stop_code, err_finale = 7.0, self.cible_parking - 64
                        else:
                            stop_code = 8.0
                            self.etape_mission = 0
                            self.prochaine_manoeuvre_autorisee = now + 10.0

            # 4. LOGIQUE CARREFOUR 
            if stop_code == 1.0:
                if self.etape_mission == 0 and now > self.prochaine_manoeuvre_autorisee:
                    if np.count_nonzero(masks[self.CLASS_DOUBLE]) > 25:
                        self.etape_mission = 1
                
                if self.etape_mission == 1:
                    # On ne détecte le piéton que si on n'est pas déjà en train de sortir d'un carrefour
                    if np.count_nonzero(masks[self.CLASS_PIETON][105:, :]) > 50:
                        stop_code, self.timer_arret, self.etape_mission = 2.0, now, 2
                
                elif self.etape_mission == 2:
                    stop_code = 2.0
                    if (now - self.timer_arret) > 1.5:
                        self.etape_mission, self.temps_traversée = 3, 0

            self.pub.publish(Point(float(np.clip(err_finale, -55, 55)), float(facteur_vitesse), stop_code))
            
        except Exception as e:
            pass

if __name__ == '__main__':
    DetecteurRoute(); rospy.spin()