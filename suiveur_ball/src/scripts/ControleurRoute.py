#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from geometry_msgs.msg import Twist, Point

class ControleurRoute:
    def __init__(self):
        rospy.init_node('controleur_route_node', anonymous=False)
        self.rate = rospy.Rate(10)
        self.Kp = 0.022 
        self.VIT_AVANCE = 0.12 
        self.statut = 1 
        self.erreur = 0
        self.facteur_vitesse = 1.0
        self.etat_p = 0
        self.t_seq = 0
        
        self.velocity = Twist()
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/road_error', Point, self.callback)
        rospy.on_shutdown(self.stop)

    def callback(self, msg):
        self.erreur = msg.x
        self.facteur_vitesse = msg.y
        self.statut = int(msg.z)

    def stop(self):
        self.pub.publish(Twist())

    def run(self):
        rospy.loginfo("--- Controleur LANCE  ---")
        while not rospy.is_shutdown():
            now = time.time()
            
            #  SEQUENCE PARKING AUTOMATIQUE (Statut 8) 
            if self.statut == 8 or self.etat_p > 0:
                if self.etat_p == 0: 
                    self.etat_p, self.t_seq = 1, now
                
                if self.etat_p == 1: # Entree
                    if now - self.t_seq < 2.6: 
                        self.velocity.linear.x, self.velocity.angular.z = 0.10, 0.35
                    else: self.etat_p, self.t_seq = 2, now
                
                elif self.etat_p == 2: # Pause
                    self.velocity.linear.x, self.velocity.angular.z = 0, 0
                    if now - self.t_seq > 5.0: self.etat_p, self.t_seq = 3, now
                
                elif self.etat_p == 3: # Recul
                    if now - self.t_seq < 2.6: 
                        self.velocity.linear.x, self.velocity.angular.z = -0.10, -0.35
                    else: self.etat_p, self.t_seq = 4, now
                
                elif self.etat_p == 4: # Sortie
                    if now - self.t_seq < 1.7: 
                        self.velocity.linear.x, self.velocity.angular.z = 0, 1.1
                    else: self.etat_p = 0

            #  ARRET PIETON / CARREFOUR (Statut 2) 
            elif self.statut == 2:
                self.velocity.linear.x, self.velocity.angular.z = 0, 0

            #  ALIGNEMENT PARKING (Statut 7) 
            elif self.statut == 7:
                self.velocity.linear.x = 0.07 * self.facteur_vitesse
                self.velocity.angular.z = max(min(-self.Kp * self.erreur, 0.8), -0.8)

            #  CONDUITE NORMALE 
            else:
                self.velocity.linear.x = self.VIT_AVANCE * self.facteur_vitesse
                cmd_rot = -self.Kp * self.erreur
                self.velocity.angular.z = max(min(cmd_rot, 1.2), -1.2)

            self.pub.publish(self.velocity)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        ControleurRoute().run()
    except rospy.ROSInterruptException:
        pass