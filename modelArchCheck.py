from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Go inside the Sequential wrapper
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {type(layer).__name__}")

#https://universe.roboflow.com/hyeonchul-jung/lying-person
#https://universe.roboflow.com/itenas-jbtoe/sitting-q96qq
#https://universe.roboflow.com/welcome-rpojx/sitting-dbnbk
#https://universe.roboflow.com/rr-qhwfw/sitting-x9rcm
#https://universe.roboflow.com/hook-works/sitting-posture-3sqgz
#https://universe.roboflow.com/student-project-phcgp/person-j8mjg
#https://universe.roboflow.com/abner/person-hgivm
#https://universe.roboflow.com/moaaz-hd0ua/person-t0qcq
#https://universe.roboflow.com/behavior-reg/standing-sitting
#https://universe.roboflow.com/lying-glihm/standing-lying