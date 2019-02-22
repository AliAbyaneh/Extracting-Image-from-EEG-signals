with open('WordReport-v1.0.txt') as f:
    lines = f.readlines()
f = open("dataDict.txt", "a")
for i in range(len(lines)):
    string = lines[i][lines[i].find('n0') : len(lines[i]) - 1] + "," + str(i) + '\n'
    f.write(string)
