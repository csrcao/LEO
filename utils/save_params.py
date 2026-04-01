from aurora.modeling_aurora import AuroraForPrediction

model = AuroraForPrediction.from_pretrained("/home/Aurora/checkpoints/Aurora_Multi_Modal_First_Version")

model.save_pretrained("/home/Aurora/checkpoints/Aurora_Release_Version")
