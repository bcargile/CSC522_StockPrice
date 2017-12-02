

infile = open("cik_ticker.csv", 'r')

contents = infile.read()
lines = contents.split('\n')

cik_ticker = {}

for l in lines:
	data = l.split('|')
	if len(data) > 2: 
		cik_ticker[data[0]] = data[1]

	
out = open("ticker_symbol.txt", 'w')	
cik2015data = open("2015.txt", 'r')
default = "Unknown Symbol"
contents = cik2015data.read()
lines = contents.split('\r')	
for l in lines:
	data = l.split('\t')
	symbol = cik_ticker.get(data[0], default)
	if len(data) > 2:
		if symbol != default:
			out.write(data[0] + "\t" + symbol + '\t' + data[2]+ '\n')
	
cik2016data = open("2016.txt", 'r')

contents = cik2016data.read()
lines = contents.split('\r')	
for l in lines:
	data = l.split('\t')
	symbol = cik_ticker.get(data[0], default)
	if len(data) > 2:
		if symbol != default:
			out.write(data[0] + "\t" + symbol + '\t' + data[2]+ '\n')

cik2017data = open("2017.txt", 'r')

contents = cik2017data.read()
lines = contents.split('\r')	
for l in lines:
	data = l.split('\t')
	symbol = cik_ticker.get(data[0], default)
	if len(data) > 2:
		if symbol != default:
			out.write(data[0] + "\t" + symbol + '\t' + data[2]+ '\n')