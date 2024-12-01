from roboflow import Roboflow
rf = Roboflow(api_key="api_key")
project = rf.workspace("projects-8f38g").project("player-detection-b6ww5") # final model
version = project.version(2)
dataset = version.download("yolov11")


# jersey model dataset url
# https://universe.roboflow.com/fastdeploy/-923m4/dataset/1

# player referee detection dataset url
# https://universe.roboflow.com/ravirajsinh-dabhi-6mq2l/ice-hockey-drjvv/dataset/2

