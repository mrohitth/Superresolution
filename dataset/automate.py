import subprocess
from pandas import read_csv

df = read_csv('data.csv')
data = (df.drop(columns = 'Case')).values
values = df["Case"].values


# def automate(geometry_number, data, batch_size):
#     for i in range(0, len(values), batch_size):
#         batch_data = data[i:i+batch_size]
#         batch_values = values[i:i+batch_size]
#         for j in range(len(batch_values)):
#             if(batch_values[j]==geometry_number):
#                 cmd = 'python3 case{0}.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}'.format(geometry_number, batch_data[j][0], batch_data[j][1], batch_data[j][2], batch_data[j][3], batch_data[j][4], batch_data[j][5], batch_data[j][6], batch_data[j][7], batch_data[j][8], batch_data[j][9], batch_data[j][10], batch_data[j][11], batch_data[j][12])
#                 subprocess.run(cmd, shell=True)


def automate(geometry_number, data, batch_size):
    for i in range(0, len(values), batch_size):
        batch_data = data[i:i+batch_size]
        batch_data_str = [[str(e) for e in row] for row in batch_data] # convert to str
        batch_values = values[i:i+batch_size]
        processes = []
        for j in range(len(batch_values)):
            if batch_values[j] == geometry_number:
                cmd = ['python3', 'case{0}.py'.format(geometry_number)]
                cmd.extend(batch_data_str[j])
                processes.append(subprocess.Popen(cmd))
        for p in processes:
            p.wait()


automate(1, data, 10)
automate(2, data, 10)
automate(3, data, 10)
automate(4, data, 10)
automate(5, data, 10)



        
 