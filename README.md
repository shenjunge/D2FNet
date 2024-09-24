# D2FNet：Dual-Domain Fusion Network for Bearing Fault Diagnosis
In this paper, we present a novel dual-domain fusion network elaborately designed for fault diagnosis. Our innovative approach integrates both time-domain and time-frequency features to enhance the discriminative capabilities of fault diagnosis. To achieve this, we introduce attention mechanisms into both the time-domain and frequency-domain feature extraction networks, benefiting the network to learn highly discernible features crucial for fault diagnosis. Specifically, we incorporate the Inception-like Frequency Map(IFMs) transformation to convert one-dimensional time-domain data into two-dimensional grayscale images efficiently. This transformation facilitates the extraction of fault-awareness features. To further boost separability in the time-frequency domain and improve diagnostic accuracy, we employ a contrastive learning-based Vision Transformer (ViT) algorithm for time-frequency feature extraction network. Through the integration of contrastive learning, self-attention mechanisms, and global feature extraction capabilities, this adaptation significantly improves the network’s proficiency in distinguishing similar categories within the time-frequency domain. 
# The overall framework
![image](https://github.com/user-attachments/assets/b931004a-d7ba-411c-a19d-da52cd454e91)
# Data processing method (a)IFMs transform (b)Wavelet transform
![image](https://github.com/user-attachments/assets/7088132b-9b1b-4087-882e-e347a5d27620)
# Comparison with different models
![image](https://github.com/user-attachments/assets/4667dbf7-7349-40ae-912e-0d0149f4eb76)
# The accuracy of the proposed model with different numbers of transformer encoders
![image](https://github.com/user-attachments/assets/6d3eac2d-3bd8-4be5-aaa4-85f53071e221)


