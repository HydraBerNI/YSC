import YSC_Project

test = YSC_Project.YSC(model_file='file', input_path='./train/inputs', output_path='./train/outputs', valid_path='./validation')
test.run(times=100,epochs=20,batch_size=5)