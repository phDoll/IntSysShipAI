# ShipzAI
ShipzAI is a reinforcment learning for the classic game Battleships.

## Instructions 
ShipzAI uses the [OpenAI Gym toolkit](https://gym.openai.com/) and [Numpy](https://numpy.org/)
for the implementation.    
To install the dependencies typ:  
```pip install gym numpy tensorflow==1.13.2 stable-baselines```  
After installtion, ShipzAI can be run like:  
```python Play.py ```  
Model can be trained with e.g.:  
```python TrainACKTR.py```
The progress of the training can be observed with Tensorboard:  
```tensorboard --port 6004 --logdir ./logs/progress_tensorboard/```

## Authors
Philipp Doll MatrNr: 700911   
Christian Leich MatNr: 699570     
Max van Aerssen MatrNr: 699795   
Group-ID: 11