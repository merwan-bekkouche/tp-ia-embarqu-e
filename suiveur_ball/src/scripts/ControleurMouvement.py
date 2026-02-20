#!/usr/bin/env python3
import rospy
import time
from geometry_msgs.msg import Twist, Point

class ControleurRoute:
    def __init__(self):
        rospy.init_node('controleur_route_node', anonymous=False)
        self.rate = rospy.Rate(10)
        
        self.Kp = 0.015
        self.VIT_AVANCE = 0.12
        self.VIT_BOOST = 0.22 
        
        self.erreur_route = 0
        self.erreur_jaune = 0
        self.statut = -1 
        
        # --- ETATS ---
        self.mode_depassement = False
        self.fin_depassement = 0
        
        self.boost_actif = False
        self.fin_boost = 0
        
        # VIRAGE DROITE
        self.force_droite_active = False
        self.fin_force_droite = 0

        self.en_arret = False
        self.temps_debut_arret = 0
        self.duree_arret = 0
        self.fin_immunite = 0 
        
        self.velocity = Twist()
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/road_error', Point, self.callback)
        
        rospy.on_shutdown(self.stop)
        print("--- CONTROLEUR PRET (Reglage Souple) ---")

    def callback(self, msg):
        self.erreur_route = msg.x
        self.erreur_jaune = msg.y
        self.statut = int(msg.z)

    def stop(self):
        self.pub.publish(Twist())

    def run(self):
        while not rospy.is_shutdown():
            now = time.time()
            
            # 1. ARRET
            if self.en_arret:
                if now - self.temps_debut_arret < self.duree_arret:
                    self.velocity.linear.x = 0.0
                    self.velocity.angular.z = 0.0
                    self.pub.publish(self.velocity)
                    self.rate.sleep()
                    continue
                else:
                    self.en_arret = False
                    self.fin_immunite = now + 2.5 
                    if self.duree_arret >= 5.0:
                        self.mode_depassement = True
                        self.fin_depassement = now + 3.0 

            # 2. DECLENCHEURS
            if now > self.fin_immunite and not self.mode_depassement:
                if self.statut == 3: 
                    self.en_arret = True; self.duree_arret = 5.0; self.temps_debut_arret = now; self.stop(); continue
                elif self.statut == 2: 
                    self.en_arret = True; self.duree_arret = 2.0; self.temps_debut_arret = now; self.stop(); continue
                
                elif self.statut == 5:
                    self.boost_actif = True
                    self.fin_boost = now + 2.0
                
                # INTERSECTION : On lance un timer court (1.0s)
                elif self.statut == 4:
                    if not self.force_droite_active: # Pour eviter le spam dans le terminal
                        print(">>> VIRAGE DROITE (1.0s) <<<")
                    self.force_droite_active = True
                    self.fin_force_droite = now + 1.0 

            # 3. TIMERS
            if self.boost_actif and now > self.fin_boost:
                self.boost_actif = False
            
            if self.force_droite_active and now > self.fin_force_droite:
                self.force_droite_active = False

            # 4. PILOTAGE
            if self.mode_depassement:
                err_active = self.erreur_jaune if self.erreur_jaune != 0 else -35
                vit_ang = -self.Kp * err_active
                self.velocity.linear.x = 0.10
                self.velocity.angular.z = max(min(vit_ang, 1.0), -1.0)
            
            # VIRAGE FORCE (Prioritaire)
            elif self.force_droite_active:
                # On ajoute un biais MODERE (+40 pixels)
                # Cela suffit a choisir la branche de droite, mais le PID continue de corriger un peu
                err_forcee = self.erreur_route + 40
                
                self.velocity.linear.x = 0.08 
                self.velocity.angular.z = max(min(-self.Kp * err_forcee, 1.5), -1.5)

            elif self.boost_actif:
                vit_ang = -self.Kp * self.erreur_route
                self.velocity.linear.x = self.VIT_BOOST
                self.velocity.angular.z = max(min(vit_ang, 1.2), -1.2)

            elif self.statut >= 1: 
                vit_ang = -self.Kp * self.erreur_route
                vitesse = self.VIT_AVANCE - (abs(self.erreur_route) * 0.001)
                if vitesse < 0.06: vitesse = 0.06
                
                self.velocity.linear.x = vitesse
                self.velocity.angular.z = max(min(vit_ang, 1.2), -1.2)
            
            else: 
                self.velocity.linear.x = 0.0
                self.velocity.angular.z = 0.0
            
            self.pub.publish(self.velocity)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        ControleurRoute().run()
    except rospy.ROSInterruptException:
        pass