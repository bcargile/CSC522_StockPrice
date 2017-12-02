infile = open("DayBefore10KRelease.txt", 'r')

contents = infile.read()
lines = contents.split('\n')
out = open("pricedatabefore.csv", 'w')
cik = 1
date = ""
symbol = ""
price = 0
for l in lines:
	data = l.split("\t")
	if len(data) > 2:
		if data[0] == "":
			price = data[4]
			out.write(cik + "\t" + symbol + "\t" + date + "\t" + price + "\n")
		else:
			
			cik = data[0]
			symbol = data[1]
			date = data[2]