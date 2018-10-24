import json

OTB2015 = json.load(open('OTB2015.json', 'r'))
videos = OTB2015.keys()

OTB2013 = dict()
for v in videos:
   if v in ['CarDark', 'Car4', 'David', 'David2', 'Sylvester', 'Trellis', 'Fish', 'Mhyang', 'Soccer', 'Matrix',
            'Ironman', 'Deer', 'Skating1', 'Shaking', 'Singer1', 'Singer2', 'Coke', 'Bolt', 'Boy', 'Dudek',
            'Crossing', 'Couple', 'Football1', 'Jogging-1', 'Jogging-2', 'Doll', 'Girl', 'Walking2', 'Walking',
            'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4', 'David3', 'Jumping', 'CarScale', 'Skiing', 'Dog1',
            'Suv', 'MotorRolling', 'MountainBike', 'Lemming', 'Liquor', 'Woman', 'FaceOcc1', 'FaceOcc2',
            'Basketball', 'Football', 'Subway', 'Tiger1', 'Tiger2']:
        OTB2013[v] = OTB2015[v]


json.dump(OTB2013, open('OTB2013.json', 'w'), indent=2) 

