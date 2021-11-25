
------------------------------------------------------------------------------------
# Experiment: TadGAN: Time Series Anomaly Detection with Generative Adversarial Networks
------------------------------------------------------------------------------------

***Unsupervised Learning:***

We would set a threshold that limits the amount of false positives to a manageable degree, and captures the most anomalous data points.

- Data preprocess 

- Training Model with unlabeled data

- Inference on unseen data by setting a threshold

- Scoring for detecting anomalies in OBD sensor data


------------------------------------------------------------------------------------
# Usage
------------------------------------------------------------------------------------

```
    python3 -m venv myenv

    source myenv/bin/activate

    pip3 --no-cache-dir install -r requirements.txt  

    - or for ubuntu

    apt-get install -y python3-venv

    python3 -m venv awesome_venv

    source awesome_venv/bin/activate

    pip3 --no-cache-dir install -r requirements.txt  

    python data-preprocess.py
    
    python train.py

    python inference.py

```

![](image/secondly_data.png)

![](image/anomaly-detection.png)


------------------------------------------------------------------------------------
