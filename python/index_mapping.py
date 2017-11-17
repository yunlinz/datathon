import csv
import numpy as np
from fuzzywuzzy import fuzz

def cleanup_zipcode(zipcode):
  if (len(zipcode) == 5):
    return zipcode
  elif (len(zipcode) < 5):
    zipcode = "0" + zipcode
    while (len(zipcode) < 5):
      zipcode = "0" + zipcode
  else:
    zipcode = zipcode[0:5]
  return zipcode


def find_inst(college_list, data):
  zipcode = cleanup_zipcode(data[-1])
  state = data[-2].upper()
  city = data[-3].upper()
  # print ("FINDING")
  # print (data)
  for line in college_list:
    i_zipcode = line[5]
    i_city = line[3].upper()
    i_inst_name = line[2].upper()
    if (i_zipcode == zipcode or i_city == city):
      # print(line)
      ratio1 = fuzz.ratio(i_inst_name, data[0].upper())
      ratio2 = fuzz.ratio(data[0].upper(),i_inst_name)
      ratio = max(ratio1, ratio2)
      #print (ratio)
      if ratio > 60:
        return line[0], line[1]

  # print(data)

  return -1,-1






def index_mapping(filename, money_file):
  # data = list(csv.reader(open(filename)))
  # print(data[0])
  # data.pop(0)
  # print (len(data))

  # sorted_data = sorted(data,key=lambda l:float(l[0]), reverse=False)
  # for i in range(4):
  #   print (sorted_data[i])

  # float_sorted_data_index = [float(line[0]) for line in sorted_data]

  # unique_sorted_data_id, unique_index = np.unique(float_sorted_data_index,return_index=True)
  # print(len(unique_sorted_data_id))

  # print (unique_index)
  # unique_index_list = [int(i) for i in unique_index]

  # file2 = open("colleges_unique_id.csv", 'w')
  # for i in unique_index_list:
  #   file2.write(sorted_data[i][0])
  #   file2.write(",")
  #   file2.write(sorted_data[i][1])
  #   file2.write(",")
  #   file2.write(sorted_data[i][2])
  #   file2.write(",")
  #   file2.write(sorted_data[i][3])
  #   file2.write(",")
  #   file2.write(sorted_data[i][4])
  #   file2.write(",")
  #   zipcode = sorted_data[i][5]
  #   zipcode2 = cleanup_zipcode(zipcode)
  #   #print("%s, %s" % (zipcode, zipcode2))
  #   file2.write(zipcode2)
  #   file2.write("\n")
  # file2.close()
  

  data = list(csv.reader(open(filename)))

  money_data = list(csv.reader(open(money_file)))
  print(money_data[0])
  money_data.pop(0)
  print (len(money_data))

  sorted_data_money = sorted(money_data,key=lambda l:(l[1]), reverse=False)
  for i in range(4):
    print (sorted_data_money[i])

  inst_names = [line[1] for line in sorted_data_money]

  unique_sorted_names, unique_index = np.unique(inst_names,return_index=True)

  print (unique_index)
  unique_index_list2 = [int(i) for i in unique_index]

  index = 0
  inst_id = []
  inst_id2 = []
  unique_index_list2.append(-1)
  for i in range(len(money_data)):
    if (i == unique_index_list2[index]):
      cid, did = find_inst(data, sorted_data_money[i][1:6])
      inst_id.append(cid)
      inst_id2.append(did)
      index += 1
    else:
      inst_id.append(cid)
      inst_id2.append(did)
    #print(inst_id[-1])

  print(len(inst_id))
  file2 = open("college_financials_with_id.csv", 'w')
  for i in range(len(money_data)):
  #for i in inst_id:
    file2.write(str(inst_id[i]))
    file2.write(",")
    file2.write(str(inst_id2[i]))
    file2.write(",")
    for item in sorted_data_money[i]:
      file2.write(item)
      file2.write(",")
    file2.write("\n")
  file2.close()



  


if __name__ == '__main__':
  #index_mapping("../data/colleges.csv", "../data/college_financials.csv")
  index_mapping("../data/colleges_unique_id.csv", "../data/college_financials.csv")
