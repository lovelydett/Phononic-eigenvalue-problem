import h5py
import numpy as np
filePath = "data/myGeneartedFile.h5"
file_real_inp = 'data/inputMatrix.npz'# Dataset for testing, which contains real optimized materials
file_real_out = 'data/Solution.npz' # Same as before but outputs


root = h5py.File(filePath, 'r')
#file_out = h5py.File("data/data2D.h5","w")
inputMatrix = np.load(file_real_inp)
Solution = np.load(file_real_out)

# print("------inputMatrix.npz------")
# for key in list(inputMatrix.keys()):
#     print(key+":",end = "")
#     print(inputMatrix[key].shape)
# print("\n------Solution.npz------")
# for key in list(Solution.keys()):
#     print(key+":",end = "")
#     print(Solution[key].shape)
# # print(list(root.keys()))
# # #['random', 'random2', 'random3', 'random5']

# print(list(root.values()))
# print(len(root.values()))
# dataset_random = root["random"]
# print(list(dataset_random.keys()))
# #['x0', 'x1', 'x10', 'x100', 'x1000', 'x1001', 'x1002', 'x1003', 'x1004', 'x1005', 'x1006', 'x1007', 'x1008', 'x1009', 'x101', 'x1010', 'x1011', 'x1012', 'x1013', 'x1014', 'x1015', 'x1016', 'x1017', 'x1018', 'x1019', 'x102', 'x1020', 'x1021', 'x1022', 'x1023', 'x1024', 'x1025', 'x1026', 'x1027', 'x1028', 'x1029', 'x103', 'x1030', 'x1031', 'x1032', 'x1033', 'x1034', 'x1035', 'x1036', 'x1037', 'x1038', 'x1039', 'x104', 'x1040', 'x1041', 'x1042', 'x1043', 'x1044', 'x1045', 'x1046', 'x1047', 'x1048', 'x1049', 'x105', 'x1050', 'x1051', 'x1052', 'x1053', 'x1054', 'x1055', 'x1056', 'x1057', 'x1058', 'x1059', 'x106', 'x1060', 'x1061', 'x1062', 'x1063', 'x1064', 'x1065', 'x1066', 'x1067', 'x1068', 'x1069', 'x107', 'x1070', 'x1071', 'x1072', 'x1073', 'x1074', 'x1075', 'x1076', 'x1077', 'x1078', 'x1079', 'x108', 'x1080', 'x1081', 'x1082', 'x1083', 'x1084', 'x1085', 'x1086', 'x1087', 'x1088', 'x1089', 'x109', 'x1090', 'x1091', 'x1092', 'x1093', 'x1094', 'x1095', 'x1096', 'x1097', 'x1098', 'x1099', 'x11'



# dataset_random_x0 = dataset_random["x0"]
# print(dataset_random_x0)
# #<HDF5 dataset "x0": shape (15360, 128, 128, 5), type "<f8">

# print(dataset_random_x0[1][1][1])
# # #[1.25382873e+11 1.92587310e+03 1.69443199e-01 3.98504967e-01
# #  1.91700555e-01]


for key in list(root.keys()):
    dataset = root[key]
    print("\n------in /root/"+key+"------")
    for key in list(dataset.keys()):
        print(key+":",end = "")
        print(dataset[key])

# with h5py.File("data/myGeneartedFile.h5", "w") as f:
#     for key in list(root.keys()):
#         dataset = root[key]
#         subset = f.create_group(key)
#         print("\n------in /root/"+key+"------")
#         for key in list(dataset.keys()):
#             XorY = subset.create_dataset(key,data=dataset[key])
#             print(key+" was create:",end = "")
#             print(XorY.shape)
# f.close()