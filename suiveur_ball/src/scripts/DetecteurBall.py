#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from ultralytics import YOLO

class DetecteurBall:
    def __init__(self):
        # Init ROS
        if not rospy.core.is_initialized():
            rospy.init_node('detecteur_ball_node', anonymous=False)

        self.bridge = CvBridge()
        
        # --- CHARGEMENT DU MODELE ---
        # Verifie que c'est bien le modele balle ici
        self.model_path = "/home/ubuntu/catkin_ws/src/suiveur_ball/src/scripts/best.onnx" 
        
        try:
            # task='detect' pour les boites
            self.model = YOLO(self.model_path, task='detect') 
            rospy.loginfo("--- MODELE BALLE CHARGE (Format 128px) ---")
        except Exception as e:
            rospy.logerr(f"Erreur chargement modele: {e}")
            exit(1)

        self.sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher('/ball_pos', Point, queue_size=1)
        
        # CORRECTION ICI : On passe a 128 car ton modele attend 128
        self.imsize = 128 
        self.CLASS_BALLE = 0 

    def camera_callback(self, msg):
        try:
            # 1. Conversion Image
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: return
            
            # On redimensionne manuellement a 128 pour eviter l'erreur ONNX
            img = cv2.resize(img, (self.imsize, self.imsize)) 
            rows, cols, _ = img.shape
            center_x_image = cols // 2

            # 2. Prediction YOLO
            # ON FORCE imgsz=128
            results = self.model.predict(img, imgsz=self.imsize, conf=0.15, verbose=False, device='cpu')
            
            ball_found = False
            best_box = None
            max_area = 0
 
            # 3. Analyse des Boites
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls.cpu())
                    
                    if cls == self.CLASS_BALLE:
                        # xywh = [x_centre, y_centre, largeur, hauteur]
                        xywh = box.xywh[0].cpu().numpy()
                        x, y, w, h = xywh
                        area = w * h
                        
                        if area > max_area:
                            max_area = area
                            best_box = xywh
                            ball_found = True

            # 4. Envoi des infos
            msg_out = Point()
            
            if ball_found:
                # Calcul erreur
                err_x = best_box[0] - center_x_image
                size_w = best_box[2] 

                msg_out.x = float(err_x)    # Erreur direction
                msg_out.y = float(size_w)   # Taille
                msg_out.z = 1.0             # Vu
                
                # --- DESSIN ---
                x, y, w, h = best_box
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
                
            else:
                msg_out.z = -1.0 # Pas de balle

            self.pub.publish(msg_out)

            # Affichage (Image toute petite 128x128 mais rapide)
            cv2.imshow("Suivi Balle", img)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(str(e))

if __name__ == '__main__':
    try:
        DetecteurBall()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass