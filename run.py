from data import TUMData

path = "tum/dataset-corridor1_512_16/mav0/imu0/data.csv"

tumdata = TUMData(path)
print(tumdata.get_accl_dataIdx(5))