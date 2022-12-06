import mmcv

datainfo = mmcv.load('/home/achieve-itn/PhD/Code/ObjDet/open-mmlab/Li3DeTr'
                     '/data/kitti/kitti_infos_train.pkl')
car_count = 0
ped_count = 0
cyc_count = 0

for data in datainfo:
    objects = list(data['annos']['name'])
    car_count += objects.count('Car')
    ped_count += objects.count('Pedestrian')
    cyc_count += objects.count('Cyclist')

print(f'Cars: {car_count}')
print(f'Ped: {ped_count}')
print(f'Cyc: {cyc_count}')

