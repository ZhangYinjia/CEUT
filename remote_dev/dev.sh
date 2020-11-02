#!/bin/bash

S1_IP=********** 
S1_USERNAME=yinjia
S2_IP=*********
S2_USERNAME=yinjia

# step.1 upload config.py and s1_clustering_server/kS1.py to
# S1_IP:/home/S1_USERNAME/encryption_clustering/clustering_server/

scp config.py $S1_USERNAME@$S1_IP:/home/$S1_USERNAME/encryption_clustering/clustering_server/
scp ../s1_clustering_server/kS1.py $S1_USERNAME@$S1_IP:/home/$S1_USERNAME/encryption_clustering/clustering_server/
scp ../s1_clustering_server/rsS1.py $S1_USERNAME@$S1_IP:/home/$S1_USERNAME/encryption_clustering/clustering_server/
scp ../s1_clustering_server/help.py $S1_USERNAME@$S1_IP:/home/$S1_USERNAME/encryption_clustering/clustering_server/

# step.2 upload config.py and s2_decryption_server/kS2.py to
# S2_IP:/home/S2_USERNAME/encryption_clustering/decryption_server/

scp config.py $S2_USERNAME@$S2_IP:/home/$S2_USERNAME/encryption_clustering/decryption_server/
scp ../s2_decryption_server/kS2.py $S2_USERNAME@$S2_IP:/home/$S2_USERNAME/encryption_clustering/decryption_server/
scp ../s2_decryption_server/rsS2.py $S2_USERNAME@$S2_IP:/home/$S2_USERNAME/encryption_clustering/decryption_server/
scp ../s2_decryption_server/help.py $S2_USERNAME@$S2_IP:/home/$S2_USERNAME/encryption_clustering/decryption_server/

# step.3 upload data file to /home/$S1_USERNAME/encryption_clustering/
scp ../data/* $S1_USERNAME@$S1_IP:/home/$S1_USERNAME/encryption_clustering/data/


